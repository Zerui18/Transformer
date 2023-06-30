import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
import pytorch_lightning as pl
from transformers import EncoderDecoderConfig, EncoderDecoderModel
from transformers import BertConfig
from dataclasses import dataclass

### CONFIG ###

@dataclass
class HFEncoderDecoderConfig:
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

	def make_hf_config(self):
		''' Create a HuggingFace `EncoderDecoderConfig` object from the current config. '''
		encoder_config = BertConfig(
			vocab_size=self.src_vocab_size,
			hidden_size=self.emb_dim,
			num_hidden_layers=self.n_blocks,
			num_attention_heads=self.n_heads,
			intermediate_size=self.emb_dim * 4,
			hidden_dropout_prob=self.dropout,
			attention_probs_dropout_prob=self.dropout,
			max_position_embeddings=self.max_len + 2,
			pad_token_id=self.pad_index,
			position_embedding_type='absolute')
		decoder_config = BertConfig(
			vocab_size=self.tgt_vocab_size,
			hidden_size=self.emb_dim,
			num_hidden_layers=self.n_blocks,
			num_attention_heads=self.n_heads,
			intermediate_size=self.emb_dim * 4,
			hidden_dropout_prob=self.dropout,
			attention_probs_dropout_prob=self.dropout,
			max_position_embeddings=self.max_len + 2,
			pad_token_id=self.pad_index,
			position_embedding_type='absolute')
		config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)
		config.decoder_start_token_id = 1
		config.eos_token_id = 2
		config.pad_token_id = self.pad_index
		return config

@dataclass
class TransformerInputBatch:
	''' A batch of training data for the Transformer model. '''
	x_src: torch.Tensor
	x_tgt: torch.Tensor
	x_src_mask: torch.Tensor
	x_tgt_mask: torch.Tensor
	y_tgt: torch.Tensor

### MODEL ###

class HFEncoderDecoder(pl.LightningModule):

	def __init__(self, config: HFEncoderDecoderConfig):
		super().__init__()
		self.save_hyperparameters()
		self.config = config
		self.learning_rate = config.learning_rate
		self.attention_type = config.attention_type
		self.criterion = nn.CrossEntropyLoss(ignore_index=config.pad_index)
		# setup sample input for tracing
		self.example_input_array = (torch.zeros(1, config.max_len, dtype=torch.long),
			      					torch.zeros(1, config.max_len, dtype=torch.long),
									torch.zeros(1, 1, config.max_len, dtype=torch.long),
									torch.zeros(1, 1, config.max_len, dtype=torch.long))
		# model parts
		self.model = EncoderDecoderModel(config.make_hf_config())

	def forward(self, src: Tensor, tgt: Tensor, src_tok_mask: Tensor, tgt_tok_mask: Tensor):
		''' Forward pass through the model.'''
		# squeeze the -2 dim of the masks
		src_tok_mask = src_tok_mask.squeeze(-2)
		tgt_tok_mask = tgt_tok_mask.squeeze(-2)
		output = self.model(input_ids=src, decoder_input_ids=tgt, attention_mask=src_tok_mask, decoder_attention_mask=tgt_tok_mask)
		return output.logits
	
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
		# masked accuracy
		non_padding_idx = y_true != self.config.pad_index
		y_pred = y_pred[non_padding_idx]
		y_true = y_true[non_padding_idx]
		accuracy = (y_pred == y_true).float().mean()
		# log the metrics
		self.log(f'{prefix}_accuracy', accuracy)

	def training_step(self, batch: TransformerInputBatch, batch_idx: int):
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
			# yield f'{idx_next[0]}: {token} \n'
			yield token
			# stop if the last token is the EOS token
			if token == eos_idx:
				break
		return tgt[0]
