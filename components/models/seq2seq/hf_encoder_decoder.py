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

	Properties:
		1. criterion: nn.CrossEntropyLoss  loss function.
		2. model: EncoderDecoderModel  the HuggingFace model.
		3. max_len: int  maximum sequence length.
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
				 metrics: dict[str, list[BaseMetric]] | None = None):
		''' Initialize the HFEncoderDecoder model.

		Args:
			1. max_len: int  maximum sequence length.
			2. src_vocab_size: int  source vocabulary size.
			3. tgt_vocab_size: int  target vocabulary size.
			4. n_blocks: int  number of transformer layers.
			5. n_heads: int  number of attention heads.
			6. emb_dim: int  embedding/hidden dimension D.
			7. dropout: float  dropout rate.
			8. bias: bool  (unused, kept for interface compatibility).
			9. weight_tying: bool  (unused, HF handles internally).
			10. use_grad_ckpt: bool  (unused, HF handles internally).
			11. pad_index: int  padding token index.
			12. attention_type: str  (unused, HF uses its own attention).
			13. optimizer: dict[str, Any]  optimizer config with 'cls' key and kwargs.
			14. metrics: dict[str, list[BaseMetric]] | None  stage-keyed metrics.
		'''
		super().__init__(optimizer=optimizer, metrics=metrics)
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
			1. src: Tensor  [int64, (B, Ts)] source token ids.
			2. tgt: Tensor  [int64, (B, Tt)] target token ids.
			3. src_tok_mask: Tensor  [bool, (B, Ts)] source attention mask.
			4. tgt_tok_mask: Tensor  [bool, (B, Tt)] target attention mask.
		Returns:
			logits: Tensor  [float32, (B, Tt, V)] vocabulary logits.
		'''
		output = self.model(input_ids=src, decoder_input_ids=tgt,
							attention_mask=src_tok_mask, decoder_attention_mask=tgt_tok_mask)
		return output.logits

	# --- BaseModel contract ---

	def produce(self, batch: dict[str, Any], requested: set[str]) -> dict[str, Any]:
		''' Produce requested outputs from a batch.

		Args:
			1. batch: dict[str, Any]  keys: x_src, x_tgt, x_src_mask, x_tgt_mask, y_tgt.
			2. requested: set[str]  subset of supports().
		Returns:
			outputs: dict[str, Any]  requested outputs.
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
			keys: set[str]  {'loss', 'y_pred', 'y_true'}.
		'''
		return {'loss', 'y_pred', 'y_true'}

	# --- Translation (model-specific) ---

	@torch.inference_mode()
	def translate(self, src: Tensor, bos_idx: int, eos_idx: int,
				  temperature: float = 1.0, max_new_tokens: int = 1000) -> Generator[int, None, None]:
		''' Translate a source sequence using greedy decoding.

		Args:
			1. src: Tensor  [int64, (Ts,)] source token ids (unbatched).
			2. bos_idx: int  BOS token id.
			3. eos_idx: int  EOS token id.
			4. temperature: float  softmax temperature.
			5. max_new_tokens: int  max tokens to generate.
		Yields:
			token: int  next generated token id.
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
