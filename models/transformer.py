import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
import pytorch_lightning as pl
from modules.transformer import TransformerEncoder, TransformerDecoder, TransformerLMHead
from modules.embedding import PosNTokEmbedding
from dataclasses import dataclass
from pytorch_lightning.utilities import grad_norm

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
    attention_type: str = 'vanilla'

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
		self.attention_type = config.attention_type
		self.criterion = nn.CrossEntropyLoss(ignore_index=config.pad_index)
		# setup sample input for tracing
		self.example_input_array = (torch.zeros(1, config.max_len, dtype=torch.long),
			      					torch.zeros(1, config.max_len, dtype=torch.long),
									torch.zeros(1, config.max_len, dtype=torch.long),
									torch.zeros(1, config.max_len, dtype=torch.long))
		# model parts
		self.src_embeddings = PosNTokEmbedding(config.src_vocab_size, config.emb_dim, config.max_len)
		self.tgt_embeddings = PosNTokEmbedding(config.tgt_vocab_size, config.emb_dim, config.max_len)
		self.encoder = TransformerEncoder(config.n_blocks, config.n_heads, config.emb_dim, config.dropout, config.bias, config.use_grad_ckpt, config.attention_type)
		self.decoder = TransformerDecoder(config.n_blocks, config.n_heads, config.emb_dim, config.dropout, config.bias, config.use_grad_ckpt, config.attention_type)
		self.lm_head = TransformerLMHead(config.emb_dim, config.tgt_vocab_size)
		# weight tying
		if config.weight_tying:
			self.tgt_embeddings.token_embedding_table.weight = self.lm_head.logits_head.weight

	def forward(self, src: Tensor, tgt: Tensor, src_tok_mask: Tensor, tgt_tok_mask: Tensor):
		''' Forward pass through the model.'''
		enc = self.encoder(self.src_embeddings(src), src_tok_mask)
		dec = self.decoder(enc, self.tgt_embeddings(tgt), src_tok_mask, tgt_tok_mask)
		logits = self.lm_head(dec)
		return logits
	
	### UTILS ###

	def calculate_loss(self, y_pred: Tensor, y_true: Tensor, prefix: str):
		''' Calculate and log loss for a batch of predictions.'''
		B, T = y_true.shape
		loss = self.criterion(y_pred.view(B * T, -1), y_true.reshape(B * T))
		# log the loss
		return {
			f'{prefix}_loss': loss
		}

	def calculate_metrics(self, y_pred: Tensor, y_true: Tensor, prefix: str):
		''' Calculate and log [accuracy] for a batch of predictions.'''
		B, T = y_true.shape
		# flatten the tensors
		y_pred = y_pred.view(B * T, -1).argmax(dim=-1)
		y_true = y_true.reshape(B * T)
		# calculate the metrics
		accuracy = (y_pred == y_true).float().mean()
		# log the metrics
		return {
			f'{prefix}_accuracy': accuracy
		}
	
	### STEPS ###

	def training_step(self, batch: TransformerInputBatch, batch_idx: int):
		y_pred = self(batch.x_src, batch.x_tgt, batch.x_src_mask, batch.x_tgt_mask)
		metrics = self.calculate_metrics(y_pred, batch.y_tgt, 'train')
		loss = self.calculate_loss(y_pred, batch.y_tgt, 'train')
		logs = {**metrics, **loss}
		self.log_dict(logs, prog_bar=True)
		self.train_losses.append(loss['train_loss'].item())
		return loss['train_loss']

	def validation_step(self, batch: TransformerInputBatch, batch_idx: int):
		y_pred = self(batch.x_src, batch.x_tgt, batch.x_src_mask, batch.x_tgt_mask)
		metrics = self.calculate_metrics(y_pred, batch.y_tgt, 'val')
		loss = self.calculate_loss(y_pred, batch.y_tgt, 'val')
		logs = {**metrics, **loss}
		self.log_dict(logs, prog_bar=True)
		self.val_losses.append(loss['val_loss'].item())
		return loss['val_loss']

	def test_step(self, batch: TransformerInputBatch, batch_idx: int):
		y_pred = self(batch.x_src, batch.x_tgt, batch.x_src_mask, batch.x_tgt_mask)
		metrics = self.calculate_metrics(y_pred, batch.y_tgt, 'test')
		loss = self.calculate_loss(y_pred, batch.y_tgt, 'test')
		logs = {**metrics, **loss}
		self.log_dict(logs, prog_bar=True)
		return loss['test_loss']

	def configure_optimizers(self):
		optimizer = getattr(torch.optim, self.config.optimizer)
		return optimizer(self.parameters(), lr=self.learning_rate)
	
	### HOOKS ###
	
	def on_train_epoch_start(self):
		self.train_losses = []
	
	def on_validation_epoch_start(self):
		self.val_losses = []
	
	def on_train_epoch_end(self):
		loss = sum(self.train_losses) / len(self.train_losses)
		print(f'Epoch {self.trainer.current_epoch} train loss:', loss)

	def on_validation_epoch_end(self):
		loss = sum(self.val_losses) / len(self.val_losses)
		print(f'Epoch {self.trainer.current_epoch} val loss:', loss)
	
	def on_before_optimizer_step(self, optimizer):
		norms = grad_norm(self, 2)
		self.log_dict(norms)

	### TRANSLATION ###
	
	@torch.inference_mode()
	def translate_with_sampling(self, src: Tensor, bos_idx: int, eos_idx: int, sampling: str = 'multinomial', temperature: float = 1.0, max_new_tokens: int = 1000):
		''' Generator function that translates a source sentence into a target sentence using sampling.
		
		Input:
			`src`: Tensor<Float>[T_in] input src tensor.

		Yields:
			int: next token in the sequence.
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
			if sampling == 'multinomial':
				idx_next = torch.multinomial(probs, num_samples=1)
			elif sampling == 'argmax':
				idx_next = torch.argmax(probs, dim=-1, keepdim=True) # (1, 1)
			# append sampled index to the running sequence
			tgt = torch.cat((tgt, idx_next), dim=1) # (1, T)
			# yield the current token
			token = int(idx_next[0].cpu().numpy())
			yield token
			# stop if the last token is the EOS token
			if token == eos_idx:
				break
	
	@torch.inference_mode()
	def translate_with_beams(self, src: Tensor, bos_idx: int, eos_idx: int, beam_width: int = 16, max_new_tokens: int = 1000):
		''' Generator function that translates a source sentence into a target sentence using beam search.
		
		Input:
			`src`: Tensor<Float>[T_in] input src tensor.
		
		Yields:
			Tensor<Long>[Beam_Width] next token in the sequence for all beams.
		
		Output:
			Tensor<Float>[Beam_Width, T_out] all beams.
		'''
		# put self into eval mode
		self.eval()
		# init inputs
		src = src.to(self.device).unsqueeze(0) # (1, T)
		tgt = torch.tensor([[bos_idx]], dtype=torch.long, device=self.device) # (1, 1)
		tgt_probs = torch.ones(beam_width, dtype=torch.float, device=self.device) # (B)
		src_mask = torch.ones_like(src, dtype=torch.bool, device=self.device)
		# generate first batch of beams
		tgt_mask = torch.ones_like(tgt, dtype=torch.bool, device=self.device)
		logits = self(src[:, -self.config.max_len:], tgt, src_mask, tgt_mask) # (1, T, C)
		# get topk beams
		logits = logits[0, -1, :] # (C)
		_, topk_idx = torch.topk(logits, k=beam_width, dim=-1) # (B)
		yield topk_idx.cpu().numpy()
		# update tgt
		tgt = torch.concat((tgt.repeat(beam_width, 1), topk_idx.unsqueeze(1)), dim=1) # (B, 2)
		# repeat src & src mask
		src = src.repeat(beam_width, 1) # (B, T)
		src_mask = src_mask.repeat(beam_width, 1) # (B, T)
		# extend the beams
		for i in range(max_new_tokens-1):
			# grow tgt
			tgt = torch.concat((tgt, torch.zeros((beam_width, 1), dtype=torch.long, device=self.device)), dim=1) # (B, T)
			# update tgt mask
			tgt_mask = torch.ones_like(tgt, dtype=torch.bool, device=self.device)
			# get the predictions
			logits = self(src[:, -self.config.max_len:], tgt[:, -self.config.max_len:], src_mask, tgt_mask) # (B, T, C)
			# focus only on the last time step
			logits = logits[:, -1, :] #(B, C)
			next_probs = F.softmax(logits, dim=-1)
			# form the new beams
			joint_probs = tgt_probs.unsqueeze(1) * next_probs # (B, C)
			# get the top k beams
			topk_probs, topk_idx = torch.topk(joint_probs.flatten(), k=beam_width, dim=-1)
			topk_idx = torch.tensor(np.stack(np.unravel_index(topk_idx.cpu().numpy(), joint_probs.shape))).T # (B, 2)
			# update current beams
			for (b, idx) in topk_idx:
				tgt[b, -1] = idx
				tgt_probs[b] = topk_probs[b]
			yield tgt[:, -1].cpu().numpy()
			# stop if all the last token are the EOS token
			if torch.all(tgt == eos_idx):
				break