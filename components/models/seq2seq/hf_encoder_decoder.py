from typing import Any, Generator

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
from transformers import EncoderDecoderConfig, EncoderDecoderModel, BertConfig

from components.models.base_model import BaseModel
from components.metrics.base_metric import BaseMetric


class HFEncoderDecoder(BaseModel):
	''' HuggingFace EncoderDecoderModel wrapper conforming to the BaseModel interface.

	Uses BERT-based encoder and decoder configurations for sequence-to-sequence tasks.

	Attributes:
		criterion: ``nn.CrossEntropyLoss``: loss function.
		model: ``EncoderDecoderModel``: the HuggingFace model.
		max_len: ``int``: maximum sequence length.
	'''

	def __init__(self,
				 max_len: int,
				 src_vocab_size: int,
				 tgt_vocab_size: int,
				 n_blocks: int,
				 n_heads: int,
				 emb_dim: int,
				 dropout: float,
				 bias: bool = False,
				 weight_tying: bool = False,
				 use_grad_ckpt: bool = False,
				 pad_index: int = 3,
				 attention_type: str = 'vanilla',
				 optimizer: dict[str, Any] = {},
				 metrics: dict[str, list[BaseMetric]] | None = None,
				 epoch_metrics: list[BaseMetric] | None = None):
		''' Initialize the HFEncoderDecoder model.

		Args:
			max_len: ``int``: maximum sequence length.
			src_vocab_size: ``int``: source vocabulary size.
			tgt_vocab_size: ``int``: target vocabulary size.
			n_blocks: ``int``: number of transformer layers.
			n_heads: ``int``: number of attention heads.
			emb_dim: ``int``: embedding/hidden dimension D.
			dropout: ``float``: dropout rate.
			bias: ``bool``: (unused, kept for interface compatibility).
			weight_tying: ``bool``: (unused, HF handles internally).
			use_grad_ckpt: ``bool``: (unused, HF handles internally).
			pad_index: ``int``: padding token index.
			attention_type: ``str``: (unused, HF uses its own attention).
			optimizer: ``dict[str, Any]``: optimizer config with 'cls' key and kwargs.
			metrics: ``dict[str, list[BaseMetric]] | None``: stage-keyed metrics.
		'''
		super().__init__(optimizer=optimizer, metrics=metrics, epoch_metrics=epoch_metrics)
		self.save_hyperparameters(ignore=['metrics'])
		self.max_len = max_len
		self.criterion = nn.CrossEntropyLoss(ignore_index=pad_index)

		# build HF config
		encoder_config = BertConfig(
			vocab_size=src_vocab_size, hidden_size=emb_dim,
			num_hidden_layers=n_blocks, num_attention_heads=n_heads,
			intermediate_size=emb_dim * 4, hidden_dropout_prob=dropout,
			attention_probs_dropout_prob=dropout, max_position_embeddings=max_len + 2,
			pad_token_id=pad_index, position_embedding_type='absolute')
		decoder_config = BertConfig(
			vocab_size=tgt_vocab_size, hidden_size=emb_dim,
			num_hidden_layers=n_blocks, num_attention_heads=n_heads,
			intermediate_size=emb_dim * 4, hidden_dropout_prob=dropout,
			attention_probs_dropout_prob=dropout, max_position_embeddings=max_len + 2,
			pad_token_id=pad_index, position_embedding_type='absolute')
		hf_config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)
		hf_config.decoder_start_token_id = 1
		hf_config.eos_token_id = 2
		hf_config.pad_token_id = pad_index
		self.model = EncoderDecoderModel(hf_config)

	def forward(self, src: Tensor, tgt: Tensor,
				src_tok_mask: Tensor, tgt_tok_mask: Tensor) -> Tensor:
		''' Forward pass through the HF EncoderDecoder model.

		Args:
			src: ``Tensor[(B, Ts), int64]``: source token ids.
			tgt: ``Tensor[(B, Tt), int64]``: target token ids.
			src_tok_mask: ``Tensor[(B, Ts), bool]``: source attention mask.
			tgt_tok_mask: ``Tensor[(B, Tt), bool]``: target attention mask.
		Returns:
			``Tensor[(B, Tt, V), float32]``: vocabulary logits.
		'''
		output = self.model(input_ids=src, decoder_input_ids=tgt,
							attention_mask=src_tok_mask, decoder_attention_mask=tgt_tok_mask)
		return output.logits

	# --- BaseModel contract ---

	def produce(self, batch: dict[str, Any], requested: set[str]) -> dict[str, Any]:
		''' Produce requested outputs from a batch.

		Args:
			batch: ``dict[str, Any]``: keys: x_src, x_tgt, x_src_mask, x_tgt_mask, y_tgt.
			requested: ``set[str]``: subset of supports().
		Returns:
			``dict[str, Any]``: requested outputs.
		'''
		y_pred = self(batch['x_src'], batch['x_tgt'], batch['x_src_mask'], batch['x_tgt_mask'])
		results: dict[str, Any] = {}
		if 'loss' in requested:
			B, T = batch['y_tgt'].shape
			results['loss'] = self.criterion(y_pred.view(B * T, -1), batch['y_tgt'].reshape(B * T))
		if 'y_pred' in requested:
			results['y_pred'] = y_pred
		if 'y_true' in requested:
			results['y_true'] = batch['y_tgt']
		return results

	def supports(self) -> set[str]:
		''' Return supported output keys.

		Returns:
			``set[str]``: {'loss', 'y_pred', 'y_true'}.
		'''
		return {'loss', 'y_pred', 'y_true'}

	# --- Translation (model-specific) ---

	@torch.inference_mode()
	def translate(self, src: Tensor, bos_idx: int, eos_idx: int,
				  temperature: float = 1.0, max_new_tokens: int = 1000) -> Generator[int, None, None]:
		''' Translate a source sequence using greedy decoding.

		Args:
			src: ``Tensor[(Ts,), int64]``: source token ids (unbatched).
			bos_idx: ``int``: BOS token id.
			eos_idx: ``int``: EOS token id.
			temperature: ``float``: softmax temperature.
			max_new_tokens: ``int``: max tokens to generate.
		Yields:
			``int``: next generated token id.
		'''
		self.eval()
		src = src.to(self.device).unsqueeze(0)  # (1, Ts)
		tgt = torch.tensor([bos_idx], dtype=torch.long, device=self.device).unsqueeze(0)
		src_mask = torch.ones_like(src, dtype=torch.bool, device=self.device)
		for _ in range(max_new_tokens):
			tgt_mask = torch.ones_like(tgt, dtype=torch.bool, device=self.device)
			logits = self(src[:, -self.max_len:], tgt[:, -self.max_len:], src_mask, tgt_mask)
			logits = logits[:, -1, :] / temperature
			probs = F.softmax(logits, dim=-1)
			idx_next = torch.argmax(probs, dim=-1, keepdim=True)
			tgt = torch.cat((tgt, idx_next), dim=1)
			token = idx_next.item()
			yield token
			if token == eos_idx:
				break
