import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch import Tensor
from typing import *
import pytorch_lightning as pl
from config import TrainingConfig, TransformerConfig
from dataset import TranslationBatch, TranslationDataset
import sentencepiece as sp

GPU = torch.device('cuda')

@torch.jit.script
def new_gelu(x: Tensor):
	"""
	Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
	Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
	"""
	return 0.5 * x * (1.0 + torch.tanh(torch.sqrt(2.0 / torch.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class InputEmbeddings(nn.Module):
	''' Apply learnable token and position embeddings to input tokens. '''

	def __init__(self, config: TransformerConfig):
		super().__init__()
		self.token_embedding_table = nn.Embedding(config.vocab_size, config.emb_dim)
		self.position_embedding_table = nn.Embedding(config.block_size, config.emb_dim)
		self.register_buffer('pos_emb_index', torch.arange(config.block_size))
	
	def forward(self, x: Tensor):
		B, T = x.shape
		tok_embd = self.token_embedding_table(x)
		pos_embd = self.position_embedding_table(self.pos_emb_index[:T])
		return tok_embd + pos_embd

class MultiHeadSelfAttention(nn.Module):

	''' Multi-head self attention.
	Implements a somewhat optimized version of the self attention by combining the q, k, v projections.
	
	Inputs:
		x: input tensor of shape (B, T, C)
		mask: mask tensor of shape broadcastable to (B, T, T)
	'''

	def __init__(self, config: TransformerConfig, is_causal: bool = False):
		super().__init__()
		self.is_causal = is_causal
		self.n_heads = config.n_heads
		self.emb_dim = config.emb_dim
		# combine q, k, v projections for efficiency
		self.qkv_projection = nn.Linear(config.emb_dim, 3 * config.emb_dim, bias=config.bias)
		self.register_buffer('scale', torch.tensor(config.emb_dim ** -0.5, dtype=torch.float32))
		if is_causal:
			self.register_buffer('causal_mask', torch.tril(torch.ones(1, config.block_size, config.block_size, dtype=torch.bool)))

	def forward(self, x: Tensor, mask: Tensor):
		B, T, C = x.shape
		# proj q, k, v for all heads
		# the heads are treated as a batch dimension
		q, k, v = self.qkv_projection(x).split(self.emb_dim, dim=2)
		q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
		k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
		v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
		# compute attention
		att_weights = (q @ k.transpose(-2, -1)) / self.scale
		mask = mask.unsqueeze(1) # apply mask over all heads
		if self.is_causal:
			mask = mask & self.causal_mask[:, :T, :T]
		att_weights = att_weights.masked_fill(mask == 0, -1e9)
		att_weights = F.softmax(att_weights, dim=-1)
		y = att_weights @ v
		# combine heads
		y = y.transpose(1, 2).contiguous().view(B, T, C)
		return y

class MultiHeadCrossAttention(nn.Module):

	''' Multi-head cross attention.
	Implements a somewhat optimized version of the cross attention by combining the k, v projections.
	
	Inputs:
		x_q: query tensor of shape (B, T, C)
		x_kv: key and value tensor of shape (B, T, C)
		mask: mask tensor of shape broadcastable to (B, T, T)
	'''

	def __init__(self, config: TransformerConfig):
		super().__init__()
		self.n_heads = config.n_heads
		self.emb_dim = config.emb_dim
		self.q_projection = nn.Linear(config.emb_dim, config.emb_dim, bias=config.bias)
		# combine k, v projections for efficiency
		self.kv_projection = nn.Linear(config.emb_dim, 2 * config.emb_dim, bias=config.bias)
		self.register_buffer('scale', torch.tensor(config.emb_dim ** -0.5, dtype=torch.float32))

	def forward(self, x_q: Tensor, x_kv: Tensor, mask: Tensor):
		# proj query for all heads
		B, T, C = x_q.shape
		q = self.q_projection(x_q)
		q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
		# proj key & value for all heads
		B, T, C = x_kv.shape
		k, v = self.kv_projection(x_kv).split(self.emb_dim, dim=2)
		k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
		v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
		# compute attention
		att_weights = (q @ k.transpose(-2, -1)) / self.scale
		mask = mask.unsqueeze(1) # apply mask over all heads
		att_weights = att_weights.masked_fill(mask == 0, -1e9)
		att_weights = F.softmax(att_weights, dim=-1)
		y = att_weights @ v
		# combine heads
		y = y.transpose(1, 2).contiguous().view(B, T, C)
		return y

class FeedFoward(nn.Module):

	def __init__(self, config: TransformerConfig):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(config.emb_dim, 4 * config.emb_dim),
			nn.ReLU(),
			nn.Linear(4 * config.emb_dim, config.emb_dim),
		)
		if config.dropout:
			self.net.append(nn.Dropout(config.dropout))

	def forward(self, x: Tensor):
		return self.net(x)

class EncoderBlock(nn.Module):

	def __init__(self, config: TransformerConfig):
		super().__init__()
		self.sa_module = MultiHeadSelfAttention(config)
		self.fw_module = FeedFoward(config)
		self.ln1 = nn.LayerNorm(config.emb_dim)
		self.ln2 = nn.LayerNorm(config.emb_dim)
	
	def forward(self, src: Tensor, src_mask: Tensor):
		x = src + self.sa_module(self.ln1(src), src_mask)
		x = x + self.fw_module(self.ln2(src))
		return x

class Encoder(nn.Module):

	def __init__(self, config: TransformerConfig):
		super().__init__()
		self.input_embeddings = InputEmbeddings(config)
		self.blocks = nn.ModuleList([EncoderBlock(config) for _ in range(config.n_blocks)])
		self.use_grad_ckpt = config.use_grad_ckpt
	
	def forward(self, src: Tensor, src_mask: Tensor):
		x = self.input_embeddings(src)
		for block in self.blocks:
			if self.use_grad_ckpt:
				forward = lambda *inputs: block(*inputs)
				x = checkpoint(forward, x, src_mask, preserve_rng_state=False)
			else:
				x = block(x, src_mask)
		return x

class DecoderBlock(nn.Module):

	def __init__(self, config: TransformerConfig):
		super().__init__()
		self.sa_module = MultiHeadSelfAttention(config, is_causal=True)
		self.ca_module = MultiHeadCrossAttention(config)
		self.fw_module = FeedFoward(config)
		self.ln1 = nn.LayerNorm(config.emb_dim)
		self.ln2 = nn.LayerNorm(config.emb_dim)
		self.ln3 = nn.LayerNorm(config.emb_dim)
	
	def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor, tgt_mask: Tensor):
		x = tgt + self.sa_module(self.ln1(tgt), tgt_mask)
		x = x + self.ca_module(self.ln2(src), self.ln2(tgt), src_mask)
		x = x + self.fw_module(self.ln3(x))
		return x

class Decoder(nn.Module):

	def __init__(self, config: TransformerConfig):
		super().__init__()
		self.input_embeddings = InputEmbeddings(config)
		self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_blocks)])
		self.use_grad_ckpt = config.use_grad_ckpt
	
	def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor, tgt_mask: Tensor):
		x = self.input_embeddings(tgt)
		for block in self.blocks:
			if self.use_grad_ckpt:
				forward = lambda *inputs: block(*inputs)
				x = checkpoint(forward, src, x, src_mask, tgt_mask, preserve_rng_state=False)
			else:
				x = block(src, x, src_mask, tgt_mask)
		return x

class LMHead(nn.Module):

	def __init__(self, config: TransformerConfig):
		super().__init__()
		self.ln = nn.LayerNorm(config.emb_dim)
		self.logits_head = nn.Linear(config.emb_dim, config.vocab_size)
	
	def forward(self, x: Tensor):
		x = self.ln(x)
		x = self.logits_head(x)
		return x

class TransformerModelLN(pl.LightningModule):

	def __init__(self, model_config: TransformerConfig, training_config: TrainingConfig):
		super().__init__()
		self.save_hyperparameters()
		self.tokenizer = sp.SentencePieceProcessor(model_file=training_config.sp_model)
		self.model_config = model_config
		self.learning_rate = training_config.learning_rate
		self.batch_size = training_config.batch_size
		self.criterion = nn.CrossEntropyLoss(ignore_index=TranslationDataset.PAD_IDX)
		# setup sample input for tracing
		self.example_input_array = (torch.zeros(1, model_config.block_size, dtype=torch.long),
			      					torch.zeros(1, model_config.block_size, dtype=torch.long),
									torch.zeros(1, 1, model_config.block_size, dtype=torch.long),
									torch.zeros(1, 1, model_config.block_size, dtype=torch.long))
		# model parts
		self.encoder = Encoder(model_config)
		self.decoder = Decoder(model_config)
		self.lm_head = LMHead(model_config)
		# weight tying
		if model_config.weight_tying:
			self.decoder.input_embeddings.token_embedding_table.weight = self.lm_head.logits_head.weight

	def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor, tgt_mask: Tensor):
		enc = self.encoder(src, src_mask)
		dec = self.decoder(enc, tgt, src_mask, tgt_mask)
		logits = self.lm_head(dec)
		return logits

	def calculate_loss(self, y_pred: Tensor, y_true: Tensor, prefix: str):
		''' Calculate and log loss for a batch of predictions.'''
		B, T = y_true.shape
		loss = self.criterion(y_pred.view(B * T, -1), y_true.view(B * T))
		# log the loss
		self.log(f'{prefix}_loss', loss)
		return loss

	def calculate_metrics(self, y_pred: Tensor, y_true: Tensor, prefix: str):
		''' Calculate and log [accuracy] for a batch of predictions.'''
		B, T = y_true.shape
		# flatten the tensors
		y_pred = y_pred.view(B * T, -1).argmax(dim=-1)
		y_true = y_true.view(B * T)
		# calculate the metrics
		accuracy = (y_pred == y_true).float().mean()
		# log the metrics
		self.log(f'{prefix}_accuracy', accuracy)

	def training_step(self, batch: TranslationBatch, batch_idx: int):
		y_pred = self(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
		self.calculate_metrics(y_pred, batch.tgt, 'train')
		return self.calculate_loss(y_pred, batch.tgt, 'train')

	def validation_step(self, batch: TranslationBatch, batch_idx: int):
		y_pred = self(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
		self.calculate_metrics(y_pred, batch.tgt, 'val')
		return self.calculate_loss(y_pred, batch.tgt, 'val')

	def test_step(self, batch: TranslationBatch, batch_idx: int):
		y_pred = self(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
		self.calculate_metrics(y_pred, batch.tgt, 'test')
		return self.calculate_loss(y_pred, batch.tgt, 'test')

	def configure_optimizers(self):
		return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

	def translate(self, src: str, max_new_tokens: int):
		src = torch.tensor(self.tokenizer.encode(src), dtype=torch.long, device=self.device).unsqueeze(0)
		src_mask = (src != TranslationDataset.PAD_IDX).view(1, 1, -1)
		gen_ids = self._generate(src, src_mask, max_new_tokens).squeeze(0).cpu().numpy()
		return self.tokenizer.decode([int(i) for i in gen_ids])

	@torch.inference_mode()
	def _generate(self, src: Tensor, src_mask: Tensor, max_new_tokens: int):
		# put self into eval mode
		self.eval()
		for _ in range(max_new_tokens):
			# get the predictions
			logits = self(x[:, -self.seq_len:])
			# focus only on the last time step
			logits = logits[:, -1, :] # becomes (B, C)
			probs = F.softmax(logits, dim=-1)
			# sample from the distribution
			idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
			# append sampled index to the running sequence
			x = torch.cat((x, idx_next), dim=1) # (B, T)
		# return self to train mode
		self.train()
		return x

from torch.nn import Transformer

class TransformerModelStockLN(pl.LightningModule):

	def __init__(self, model_config: TransformerConfig, training_config: TrainingConfig):
		super().__init__()
		self.save_hyperparameters()
		self.tokenizer = sp.SentencePieceProcessor(model_file=training_config.sp_model)
		self.model_config = model_config
		self.learning_rate = training_config.learning_rate
		self.batch_size = training_config.batch_size
		self.criterion = nn.CrossEntropyLoss(ignore_index=TranslationDataset.PAD_IDX)
		# model parts
		self.input_embeddings = InputEmbeddings(model_config)
		self.transformer = Transformer(d_model=model_config.emb_dim, nhead=model_config.n_heads,
				 						num_encoder_layers=model_config.n_blocks, num_decoder_layers=model_config.n_blocks,
										dim_feedforward=model_config.emb_dim * 4, dropout=model_config.dropout,
										activation='gelu', batch_first=True, norm_first=True)
		self.lm_head = LMHead(model_config)

	def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor, tgt_mask: Tensor):
		src_emb = self.input_embeddings(src)
		tgt_emb = self.input_embeddings(tgt)
		if True:
			# for compatibility with the custom transformer
			src_mask = src_mask.squeeze(1)
			tgt_mask = tgt_mask.squeeze(1)
		emb = self.transformer(src_emb, tgt_emb, src_key_padding_mask = src_mask, tgt_key_padding_mask = tgt_mask)
		logits = self.lm_head(emb)
		return logits

	def calculate_loss(self, y_pred: Tensor, y_true: Tensor, prefix: str):
		''' Calculate and log loss for a batch of predictions.'''
		B, T = y_true.shape
		loss = self.criterion(y_pred.view(B * T, -1), y_true.view(B * T))
		# log the loss
		self.log(f'{prefix}_loss', loss)
		return loss

	def calculate_metrics(self, y_pred: Tensor, y_true: Tensor, prefix: str):
		''' Calculate and log [accuracy] for a batch of predictions.'''
		B, T = y_true.shape
		# flatten the tensors
		y_pred = y_pred.view(B * T, -1).argmax(dim=-1)
		y_true = y_true.view(B * T)
		# calculate the metrics
		accuracy = (y_pred == y_true).float().mean()
		# log the metrics
		self.log(f'{prefix}_accuracy', accuracy)

	def training_step(self, batch: TranslationBatch, batch_idx: int):
		y_pred = self(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
		self.calculate_metrics(y_pred, batch.tgt, 'train')
		return self.calculate_loss(y_pred, batch.tgt, 'train')

	def validation_step(self, batch: TranslationBatch, batch_idx: int):
		y_pred = self(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
		self.calculate_metrics(y_pred, batch.tgt, 'val')
		return self.calculate_loss(y_pred, batch.tgt, 'val')

	def test_step(self, batch: TranslationBatch, batch_idx: int):
		y_pred = self(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
		self.calculate_metrics(y_pred, batch.tgt, 'test')
		return self.calculate_loss(y_pred, batch.tgt, 'test')

	def configure_optimizers(self):
		return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

	@torch.inference_mode()
	def translate(self, src: str, temperature: float = 1.0, max_new_tokens: int = 1000):
		''' Generator function that translates a source sentence into a target sentence.'''
		# put self into eval mode
		self.eval()
		# init inputs
		src = torch.tensor(self.tokenizer.encode(src), dtype=torch.long, device=self.device).unsqueeze(0) # (1, T)
		tgt = torch.tensor([TranslationDataset.BOS_IDX], dtype=torch.long, device=self.device).unsqueeze(0) # (1, 1)
		# src_mask = torch.ones_like(src, dtype=torch.bool, device=self.device)
		for i in range(max_new_tokens):
			# update tgt mask
			# tgt_mask = torch.ones_like(tgt, dtype=torch.bool, device=self.device)
			# get the predictions
			logits = self(src[:, -self.model_config.block_size:], tgt[:, -self.model_config.block_size:], None, None) # (1, T, C)
			# focus only on the last time step
			logits = logits[:, -1, :] #(1, C)
			logits /= temperature
			probs = F.softmax(logits, dim=-1)
			# sample from the distribution
			idx_next = torch.argmax(probs, dim=-1, keepdim=True) # (1, 1)
			# append sampled index to the running sequence
			tgt = torch.cat((tgt, idx_next), dim=1) # (1, T)
			# yield the current token
			token = self.tokenizer.decode([int(idx_next[0].cpu().numpy())])
			# print(f'{i}:', idx_next[0], token)
			yield f'{idx_next[0]}: {token} \n'
			# if len(token) > 0:
			# 	yield token
			# stop if the last token is the EOS token
			if idx_next[0] == TranslationDataset.EOS_IDX:
				break
		return self.tokenizer.decode([int(t) for t in tgt[0].cpu().numpy()])