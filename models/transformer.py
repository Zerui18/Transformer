import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
import pytorch_lightning as pl
from modules.transformer import TransformerEncoder, TransformerDecoder, TransformerLMHead
from modules.embedding import PosNTokEmbedding
from dataclasses import dataclass

### CONFIG ###

@dataclass
class TransformerConfig:
    max_len: int
    src_vocab_size: int
    tgt_vocab_size: int
    n_blocks: int 
    n_heads: int 
    emb_dim: int 
    dropout: float
    bias: bool 
    weight_tying: bool
    use_grad_ckpt: bool
    pad_index: int
    optimizer: str
    learning_rate: float

### INPUT ###

@dataclass
class TransformerInputBatch:
	''' A batch of training data for the Transformer model. '''
	x_src: torch.Tensor
	x_tgt: torch.Tensor
	x_src_mask: torch.Tensor
	x_tgt_mask: torch.Tensor
	y_tgt: torch.Tensor

### MODEL ###

class Transformer(pl.LightningModule):

	def __init__(self, config: TransformerConfig):
		super().__init__()
		self.save_hyperparameters()
		self.config = config
		self.learning_rate = config.learning_rate
		self.criterion = nn.CrossEntropyLoss(ignore_index=config.pad_index)
		# setup sample input for tracing
		self.example_input_array = (torch.zeros(1, config.max_len, dtype=torch.long),
			      					torch.zeros(1, config.max_len, dtype=torch.long),
									torch.zeros(1, 1, config.max_len, dtype=torch.long),
									torch.zeros(1, 1, config.max_len, dtype=torch.long))
		# model parts
		self.src_embeddings = PosNTokEmbedding(config.src_vocab_size, config.emb_dim, config.max_len)
		self.tgt_embeddings = PosNTokEmbedding(config.tgt_vocab_size, config.emb_dim, config.max_len)
		self.encoder = TransformerEncoder(config.n_blocks, config.n_heads, config.emb_dim, config.dropout, config.bias, config.use_grad_ckpt)
		self.decoder = TransformerDecoder(config.n_blocks, config.n_heads, config.emb_dim, config.dropout, config.bias, config.use_grad_ckpt)
		self.lm_head = TransformerLMHead(config.emb_dim, config.tgt_vocab_size)
		# weight tying
		if config.weight_tying:
			self.src_embeddings.token_embedding_table.weight = self.lm_head.logits_head.weight

	def forward(self, src: Tensor, tgt: Tensor, src_tok_mask: Tensor, tgt_tok_mask: Tensor):
		''' Forward pass through the model.'''
		enc = self.encoder(self.src_embeddings(src), src_tok_mask)
		dec = self.decoder(enc, self.tgt_embeddings(tgt), src_tok_mask, tgt_tok_mask)
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

	def training_step(self, batch: TransformerInputBatch, batch_idx: int):
		# opt = self.optimizer
		# opt.zero_grad()
		y_pred = self(batch.x_src, batch.x_tgt, batch.x_src_mask, batch.x_tgt_mask)
		self.calculate_metrics(y_pred, batch.y_tgt, 'train')
		return self.calculate_loss(y_pred, batch.y_tgt, 'train')

	def validation_step(self, batch: TransformerInputBatch, batch_idx: int):
		y_pred = self(batch.x_src, batch.x_tgt, batch.x_src_mask, batch.x_tgt_mask)
		self.calculate_metrics(y_pred, batch.y_tgt, 'val')
		return self.calculate_loss(y_pred, batch.y_tgt, 'val')

	def test_step(self, batch: TransformerInputBatch, batch_idx: int):
		y_pred = self(batch.x_src, batch.x_tgt, batch.x_src_mask, batch.x_tgt_mask)
		self.calculate_metrics(y_pred, batch.y_tgt, 'test')
		return self.calculate_loss(y_pred, batch.y_tgt, 'test')

	def configure_optimizers(self):
		optimizer = getattr(torch.optim, self.config.optimizer)
		return optimizer(self.parameters(), lr=self.learning_rate)
	
	@torch.inference_mode()
	def translate(self, src: Tensor, bos_idx: int, eos_idx: int, temperature: float = 1.0, max_new_tokens: int = 1000):
		''' Generator function that translates a source sentence into a target sentence.
		
		Input:
			`src`: Tensor<Float>[T_in] input src tensor.
		
		Output:
			Tensor<Float>[T_out] output tgt tensor.
		'''
		# put self into eval mode
		self.eval()
		# init inputs
		src = src.to(self.device).unsqueeze(0) # (1, T)
		tgt = torch.tensor([bos_idx], dtype=torch.long, device=self.device).unsqueeze(0) # (1, 1)
		src_mask = torch.ones_like(src, dtype=torch.bool, device=self.device)
		for i in range(max_new_tokens):
			# update tgt mask
			tgt_mask = torch.ones_like(tgt, dtype=torch.bool, device=self.device)
			# get the predictions
			logits = self(src[:, -self.config.max_len:], tgt[:, -self.config.max_len:], src_mask, tgt_mask) # (1, T, C)
			# focus only on the last time step
			logits = logits[:, -1, :] #(1, C)
			logits /= temperature
			probs = F.softmax(logits, dim=-1)
			# sample from the distribution
			idx_next = torch.argmax(probs, dim=-1, keepdim=True) # (1, 1)
			# append sampled index to the running sequence
			tgt = torch.cat((tgt, idx_next), dim=1) # (1, T)
			# yield the current token
			token = int(idx_next[0].cpu().numpy())
			# print(f'{i}:', idx_next[0], token)
			yield f'{idx_next[0]}: {token} \n'
			# if len(token) > 0:
			# 	yield token
			# stop if the last token is the EOS token
			if token == eos_idx:
				break
		return tgt[0]