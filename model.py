import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch import Tensor
from typing import *
import pytorch_lightning as pl
import math
from dataset import TranslationBatch, TranslationDataset
from config import TransformerConfig, TrainingConfig

class InputEmbeddings(nn.Module):
	''' Apply learnable token and position embeddings to input tokens. '''

	def __init__(self, config: TransformerConfig, is_src = True):
		super().__init__()
		vocab_size = config.src_vocab_size if is_src else config.tgt_vocab_size
		self.token_embedding_table = nn.Embedding(vocab_size, config.emb_dim)
		self.position_embedding_table = nn.Embedding(config.max_len, config.emb_dim)
		self.register_buffer('pos_emb_index', torch.arange(config.max_len))
	
	def forward(self, x: Tensor):
		B, T = x.shape
		tok_embd = self.token_embedding_table(x)
		pos_embd = self.position_embedding_table(self.pos_emb_index[:T])
		return tok_embd + pos_embd

class MultiHeadSelfAttention(nn.Module):

	''' Multi-head self attention.
	Implements a somewhat optimized version of the self attention by combining the q, k, v projections.
	
	Inputs:
		`x`: Tensor<Float>[B, T, C] input tensor.
		`tok_mask`: Tensor<Bool>[B, T] per-token mask applied to the `x`, false is masked out, true is preserved - masks both keys and queries.

	Outputs:
		Tensor<Float>[B, T, C] output tensor.
	'''

	def __init__(self, config: TransformerConfig, is_causal: bool = False):
		super().__init__()
		self.is_causal = is_causal
		self.n_heads = config.n_heads
		self.emb_dim = config.emb_dim
		self.attn_dropout = nn.Dropout(config.dropout)
		self.resid_dropout = nn.Dropout(config.dropout)
		# combine q, k, v projections for efficiency
		self.qkv_projection = nn.Linear(config.emb_dim, 3 * config.emb_dim, bias=config.bias)
		# output projection
		self.c_proj = nn.Linear(config.emb_dim, config.emb_dim, bias=config.bias)
		if self.is_causal:
			self.register_buffer('causal_mask', torch.tril(torch.ones(config.max_len, config.max_len, dtype=torch.bool)))

	def forward(self, x: Tensor, tok_mask: Tensor):
		B, T, C = x.shape
		# proj q, k, v for all heads
		# the heads are treated as a batch dimension
		q, k, v = self.qkv_projection(x).split(self.emb_dim, dim=2)
		q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
		k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
		v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
		# compute attention
		att_weights = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
		mask = tok_mask.view(B, 1, 1, T) # (B, 1, 1, T) <=> (B, nh, T, T)
		if self.is_causal:
			mask = mask & self.causal_mask[None, None, :T, :T]
		att_weights = att_weights.masked_fill(mask == 0, -1e9)
		att_weights = F.softmax(att_weights, dim=-1)
		att_weights = self.attn_dropout(att_weights)
		y = att_weights @ v
		# combine heads
		y = y.transpose(1, 2).contiguous().view(B, T, C)
		y = self.resid_dropout(self.c_proj(y))
		return y

class MultiHeadCrossAttention(nn.Module):

	''' Multi-head cross attention.
	Implements a somewhat optimized version of the cross attention by combining the k, v projections.
	
	Inputs:
		`x_q`: Tensor<Float>[B, T_q, C] query input tensor.
		`x_kv`: Tensor<Float>[B, T_kv, C] key and value input tensor.
		`q_tok_mask`: Tensor<Bool>[B, T_q] mask applied to the `x_q`, false is masked out, true is preserved - applies to q only.
		`kv_tok_mask`: Tensor<Bool>[B, T_kv] mask applied to the `x_kv`, false is masked out, true is preserved - applies to k and v.

	Outputs:
		Tensor<Float>[B, T_q, C] output tensor.
	'''

	def __init__(self, config: TransformerConfig):
		super().__init__()
		self.n_heads = config.n_heads
		self.emb_dim = config.emb_dim
		self.attn_dropout = nn.Dropout(config.dropout)
		self.resid_dropout = nn.Dropout(config.dropout)
		self.q_projection = nn.Linear(config.emb_dim, config.emb_dim, bias=config.bias)
		# combine k, v projections for efficiency
		self.kv_projection = nn.Linear(config.emb_dim, 2 * config.emb_dim, bias=config.bias)
		# output projection
		self.c_proj = nn.Linear(config.emb_dim, config.emb_dim, bias=config.bias)

	def forward(self, x_q: Tensor, x_kv: Tensor, q_tok_mask: Tensor, kv_tok_mask: Tensor):
		# proj query for all heads
		B, T_q, C = x_q.shape
		q = self.q_projection(x_q)
		q = q.view(B, T_q, self.n_heads, C // self.n_heads).transpose(1, 2)
		# proj key & value for all heads
		B, T_kv, C = x_kv.shape
		k, v = self.kv_projection(x_kv).split(self.emb_dim, dim=2)
		k = k.view(B, T_kv, self.n_heads, C // self.n_heads).transpose(1, 2)
		v = v.view(B, T_kv, self.n_heads, C // self.n_heads).transpose(1, 2)
		# compute attention
		att_weights = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
		# merge masks
		q_tok_mask = q_tok_mask.unsqueeze(2) # (N, T_q, 1)
		kv_tok_mask = kv_tok_mask.unsqueeze(1) # (N, 1, T_kv)
		attn_mask = q_tok_mask & kv_tok_mask
		# apply mask
		att_weights = att_weights.masked_fill(attn_mask.unsqueeze(1) == 0, -1e9)
		att_weights = F.softmax(att_weights, dim=-1)
		att_weights = self.attn_dropout(att_weights)
		y = att_weights @ v
		# combine heads
		y = y.transpose(1, 2).contiguous().view(B, T_q, C)
		y = self.resid_dropout(self.c_proj(y))
		return y

class FeedFoward(nn.Module):

	def __init__(self, config: TransformerConfig):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(config.emb_dim, 4 * config.emb_dim),
			nn.GELU(approximate='tanh'),
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
		x = x + self.ca_module(self.ln2(tgt), self.ln2(src), tgt_mask, src_mask)
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
		self.logits_head = nn.Linear(config.emb_dim, config.tgt_vocab_size)
	
	def forward(self, x: Tensor):
		x = self.ln(x)
		x = self.logits_head(x)
		return x

import pytorch_lightning as pl

class TransformerModelLN(pl.LightningModule):

	def __init__(self, model_config: TransformerConfig, training_config: TrainingConfig):
		super().__init__()
		self.save_hyperparameters()
		# self.tokenizer = sp.SentencePieceProcessor(model_file=training_config.sp_model)
		self.model_config = model_config
		self.learning_rate = training_config.learning_rate
		self.batch_size = training_config.batch_size
		self.criterion = nn.CrossEntropyLoss(ignore_index=TranslationDataset.PAD_IDX)
		# setup sample input for tracing
		self.example_input_array = (torch.zeros(1, model_config.max_len, dtype=torch.long),
			      					torch.zeros(1, model_config.max_len, dtype=torch.long),
									torch.zeros(1, 1, model_config.max_len, dtype=torch.long),
									torch.zeros(1, 1, model_config.max_len, dtype=torch.long))
		# model parts
		self.encoder = Encoder(model_config)
		self.decoder = Decoder(model_config)
		self.lm_head = LMHead(model_config)
		# weight tying
		if model_config.weight_tying:
			self.decoder.input_embeddings.token_embedding_table.weight = self.lm_head.logits_head.weight

	def forward(self, src: Tensor, tgt: Tensor, src_tok_mask: Tensor, tgt_tok_mask: Tensor):
		enc = self.encoder(src, src_tok_mask)
		dec = self.decoder(enc, tgt, src_tok_mask, tgt_tok_mask)
		logits = self.lm_head(dec)
		return logits
	
	def calculate_loss(self, y_pred: Tensor, y_true: Tensor, prefix: str):
		''' Calculate and log loss for a batch of predictions.'''
		B, T = y_true.shape
		loss = self.criterion(y_pred.view(B * T, -1), y_true.reshape(B * T))
		# log the loss
		self.log(f'{prefix}_loss', loss)
		return loss

	def calculate_metrics(self, y_pred: Tensor, y_true: Tensor, prefix: str):
		''' Calculate and log [accuracy] for a batch of predictions.'''
		B, T = y_true.shape
		# flatten the tensors
		y_pred = y_pred.view(B * T, -1).argmax(dim=-1)
		y_true = y_true.reshape(B * T)
		# calculate the metrics
		accuracy = (y_pred == y_true).float().mean()
		# log the metrics
		self.log(f'{prefix}_accuracy', accuracy)

	def training_step(self, batch: TranslationBatch, batch_idx: int):
		# opt = self.optimizer
		# opt.zero_grad()
		y_pred = self(batch.x_src, batch.x_tgt, batch.x_src_mask, batch.x_tgt_mask)
		self.calculate_metrics(y_pred, batch.y_tgt, 'train')
		return self.calculate_loss(y_pred, batch.y_tgt, 'train')

	def validation_step(self, batch: TranslationBatch, batch_idx: int):
		y_pred = self(batch.x_src, batch.x_tgt, batch.x_src_mask, batch.x_tgt_mask)
		self.calculate_metrics(y_pred, batch.y_tgt, 'val')
		return self.calculate_loss(y_pred, batch.y_tgt, 'val')

	def test_step(self, batch: TranslationBatch, batch_idx: int):
		y_pred = self(batch.x_src, batch.x_tgt, batch.x_src_mask, batch.x_tgt_mask)
		self.calculate_metrics(y_pred, batch.y_tgt, 'test')
		return self.calculate_loss(y_pred, batch.y_tgt, 'test')

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
		src_mask = torch.ones_like(src, dtype=torch.bool, device=self.device)
		for i in range(max_new_tokens):
			# update tgt mask
			tgt_mask = torch.ones_like(tgt, dtype=torch.bool, device=self.device)
			# get the predictions
			logits = self(src[:, -self.model_config.max_len:], tgt[:, -self.model_config.max_len:], src_mask, tgt_mask) # (1, T, C)
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