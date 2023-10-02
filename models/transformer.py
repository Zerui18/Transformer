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
from metrics import get_bleu_score
from toknizers import Tokenizer

UNK_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
PAD_IDX = 3

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
	output_attention: bool = False

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

	def __init__(self, config: TransformerConfig, tokenizer: Tokenizer):
		super().__init__()
		self.save_hyperparameters()
		self.config = config
		self.learning_rate = config.learning_rate
		self.attention_type = config.attention_type
		self.criterion = nn.CrossEntropyLoss(ignore_index=config.pad_index)
		self.tokenizer = tokenizer
		# setup sample input for tracing
		self.example_input_array = (torch.zeros(1, config.max_len, dtype=torch.long),
			      					torch.zeros(1, config.max_len, dtype=torch.long),
									torch.zeros(1, config.max_len, dtype=torch.long),
									torch.zeros(1, config.max_len, dtype=torch.long))
		# model parts
		self.src_embeddings = PosNTokEmbedding(config.src_vocab_size, config.emb_dim, config.max_len)
		self.tgt_embeddings = PosNTokEmbedding(config.tgt_vocab_size, config.emb_dim, config.max_len)
		self.encoder = TransformerEncoder(config.n_blocks, config.n_heads, config.emb_dim, config.dropout, config.bias, config.use_grad_ckpt, config.attention_type, config.output_attention)
		self.decoder = TransformerDecoder(config.n_blocks, config.n_heads, config.emb_dim, config.dropout, config.bias, config.use_grad_ckpt, config.attention_type, config.output_attention)
		self.lm_head = TransformerLMHead(config.emb_dim, config.tgt_vocab_size)
		# weight tying
		if config.weight_tying:
			self.tgt_embeddings.token_embedding_table.weight = self.lm_head.logits_head.weight
		# attention hooking
		if config.output_attention:
			self.attention_weights = {}
			self.hook_attention_layers()


	def hook_attention_layers(self):
		''' Hook the attention layers to output their attention weights. '''
		def hook_attention_layer(module, input, output):
			self.attention_weights[module.tag] = output[1]
		for module in self.modules():
			if type(module).__name__ == 'MultiHeadSelfAttention' or type(module).__name__ == 'MultiHeadCrossAttention':
				module.register_forward_hook(hook_attention_layer)

	def encoder_forward(self, src: Tensor, src_tok_mask: Tensor):
		''' Forward pass through the encoder.'''
		return self.encoder(self.src_embeddings(src), src_tok_mask)
	
	def incremental_forward(self, enc: Tensor, tgt: Tensor, src_tok_mask: Tensor, tgt_tok_mask: Tensor):
		''' Forward pass through the decoder and lm head. Used in incremental decoding.'''
		dec = self.decoder(enc, self.tgt_embeddings(tgt), src_tok_mask, tgt_tok_mask)
		logits = self.lm_head(dec)
		return logits

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
	
	@torch.inference_mode()
	def generate_bleu_report(self, samples: list, references: list):
		''' Generate a BLEU report for a list of samples and their corresponding references.'''
		# put self into eval mode
		greedy_generations = []
		greedy_bleu_scores = []
		bs_generations = []
		bs_bleu_scores = []
		self.eval()
		for sample, reference in zip(samples, references):
			sample = torch.tensor(self.tokenizer.tokenize(sample), dtype=torch.long, device=self.device)
			reference = self.tokenizer.tokenize(reference)
			# greedy decoding
			greedy_translated = list(self.translate_with_sampling(sample, BOS_IDX, EOS_IDX, sampling='argmax', max_new_tokens=128))
			greedy_generations.append(self.tokenizer.detokenize(greedy_translated))
			greedy_bleu = get_bleu_score(greedy_translated, reference, self.tokenizer)
			greedy_bleu_scores.append(greedy_bleu)
			# beam search decoding
			bs_translated = [int(beams[0]) for beams in self.translate_with_beams(sample, BOS_IDX, EOS_IDX, beam_width=16, max_new_tokens=128)]
			bs_generations.append(self.tokenizer.detokenize(bs_translated))
			bs_bleu = get_bleu_score(bs_translated, reference, self.tokenizer)
			bs_bleu_scores.append(bs_bleu)
		# log results
		greedy_average_bleu = sum(greedy_bleu_scores) / len(greedy_bleu_scores)
		bs_average_bleu = sum(bs_bleu_scores) / len(bs_bleu_scores)
		self.log_dict({
			'bleu_greedy': greedy_average_bleu,
			'bleu_bs16': bs_average_bleu,
			'bleu_ave': (greedy_average_bleu + bs_average_bleu) / 2
		})
		# print generated samples
		for sample, reference, greedy, bs in zip(samples, references, greedy_generations, bs_generations):
			print('-' * 50)
			print('Step: ', self.global_step)
			print(f'Sample: {sample}')
			print(f'Refere: {reference}')
			print(f'Greedy: {greedy}')
			print(f'Beam16: {bs}')
			print()
	
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
		if self.global_step > 0:
			# generate a BLEU report
			samples_length = 32
			samples = self._val_dataloader.dataset.df.head(samples_length).src.tolist()
			references = self._val_dataloader.dataset.df.head(samples_length).tgt.tolist()
			print('Generating BLEU report...')
			print('-' * 50)
			print('Step: ', self.global_step)
			print('Samples:', samples)
			print('References:', references)
			self.generate_bleu_report(samples, references)
	
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
		assert sampling in ['multinomial', 'argmax'], f'Invalid sampling method: {sampling}'
		# put self into eval mode
		self.eval()

		### ENCODER INPUTS
		src = src.to(self.device).unsqueeze(0) # (1, T)
		src_mask = torch.ones_like(src, dtype=torch.bool, device=self.device)

		### DECODER INPUTS
		enc = self.encoder_forward(src[:, -self.config.max_len:], src_mask) # (1, T, C)
		tgt = torch.tensor([bos_idx], dtype=torch.long, device=self.device).unsqueeze(0) # (1, 1)

		### SAMPLING STAGE
		for i in range(max_new_tokens):
			# update tgt mask
			tgt_mask = torch.ones_like(tgt, dtype=torch.bool, device=self.device)
			# get the predictions
			logits = self.incremental_forward(enc, tgt[:, -self.config.max_len:], src_mask, tgt_mask) # (1, T, C)
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
		
		### ENCODER INPUTS
		src = src.to(self.device).unsqueeze(0) # (1, T)
		src_mask = torch.ones_like(src, dtype=torch.bool, device=self.device)

		### DECODER INPUTS
		enc = self.encoder_forward(src[:, -self.config.max_len:], src_mask) # (1, T, C)
		tgt = torch.tensor([[bos_idx]], dtype=torch.long, device=self.device) # (1, 1)
		tgt_mask = torch.ones_like(tgt, dtype=torch.bool, device=self.device)
		tgt_probs = torch.ones(beam_width, dtype=torch.float, device=self.device) # (B)

		### EALRY STOPPING CRITERIA
		eos_reached = torch.zeros(beam_width, dtype=torch.bool, device=self.device) # (B)

		### BEAM SEARCH STAGE 1
		# generate first batch of beams
		logits = self.incremental_forward(enc, tgt, src_mask, tgt_mask) # (1, 1, C)
		# get topk beams
		logits = logits[0, 0, :] # (C)
		_, topk_idx = torch.topk(logits, k=beam_width, dim=-1) # (B)
		eos_reached = eos_reached | (topk_idx == eos_idx)
		yield topk_idx.cpu().numpy()

		### BEAM SEARCH STAGE 1+T
		# Repeat the enc and tgt for each beam
		enc = enc.repeat(beam_width, 1, 1) # (B, T, C)
		tgt = torch.concat((tgt.repeat(beam_width, 1), topk_idx.unsqueeze(1)), dim=1) # (B, 2)

		# extend the beams
		for i in range(max_new_tokens-1):
			if torch.all(eos_reached):
				break
			# clip tgt length
			tgt = tgt[:, -self.config.max_len:]
			# update tgt mask
			tgt_mask = torch.ones_like(tgt, dtype=torch.bool, device=self.device)
			# get the predictions
			logits = self.incremental_forward(enc, tgt, src_mask, tgt_mask) # (B, T, C)
			# focus only on the last time step
			logits = logits[:, -1, :] #(B, C)
			next_probs = F.softmax(logits, dim=-1)
			# form the new beams
			joint_probs = tgt_probs.unsqueeze(1) * next_probs # (B, C)
			# get the top k beams
			topk_probs, topk_idx = torch.topk(joint_probs.flatten(), k=beam_width, dim=-1)
			topk_idx = torch.tensor(np.stack(np.unravel_index(topk_idx.cpu().numpy(), joint_probs.shape)), device=self.device).T # (B, 2)
			# update current beams
			eos_reached = eos_reached | (topk_idx[:, 1] == eos_idx)
			# grow tgt
			tgt = torch.concat((tgt, torch.zeros((beam_width, 1), dtype=torch.long, device=self.device)), dim=1) # (B, T)
			for (b, idx) in topk_idx:
				tgt[b, -1] = idx
				tgt_probs[b] = topk_probs[b]
			yield tgt[:, -1].cpu().numpy()
			# stop if all the last token are the EOS token
			if torch.all(tgt == eos_idx):
				break