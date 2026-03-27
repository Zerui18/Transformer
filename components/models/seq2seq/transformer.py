from typing import Any, Generator

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor

from components.models.base_model import BaseModel
from components.metrics.base_metric import BaseMetric
from components.tokenizers.base_tokenizer import BaseTokenizer
from components.modules.transformer import TransformerEncoder, TransformerDecoder, TransformerLMHead
from components.modules.embedding import PosNTokEmbedding

class Transformer(BaseModel):
	BOS_IDX = 1
	EOS_IDX = 2

	''' Encoder-decoder transformer for sequence-to-sequence tasks.

	Composes token+position embeddings, transformer encoder/decoder stacks,
	and a language model head. Supports weight tying, attention hooking,
	greedy decoding, and beam search.

	Attributes:
		criterion: ``nn.CrossEntropyLoss``: Loss function (ignores pad tokens).
		tokenizer: ``BaseTokenizer | None``: Tokenizer for decoding-based metrics.
		src_embeddings: ``PosNTokEmbedding``: Source token + position embeddings.
		tgt_embeddings: ``PosNTokEmbedding``: Target token + position embeddings.
		encoder: ``TransformerEncoder``: The encoder stack.
		decoder: ``TransformerDecoder``: The decoder stack.
		lm_head: ``TransformerLMHead``: Projection to vocabulary logits.
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
				 weight_tying: str | bool = False,
				 use_grad_ckpt: bool = False,
				 pad_index: int = 3,
				 attention_type: str = 'vanilla',
				 output_attention: bool = False,
				 tokenizer: BaseTokenizer | None = None,
				 optimizer: dict[str, Any] = {},
				 metrics: dict[str, list[BaseMetric]] | None = None):
		''' Initialize the Transformer model.

		Args:
			max_len: ``int``: Maximum sequence length for source and target.
			src_vocab_size: ``int``: Source vocabulary size.
			tgt_vocab_size: ``int``: Target vocabulary size.
			n_blocks: ``int``: Number of encoder/decoder blocks.
			n_heads: ``int``: Number of attention heads H.
			emb_dim: ``int``: Embedding dimension D.
			dropout: ``float``: Dropout rate.
			bias: ``bool``: Whether to use bias in attention projections.
			weight_tying: ``str | bool``: ``'3-way'``, ``'2-way'``, or ``False`` to disable.
			use_grad_ckpt: ``bool``: Enable gradient checkpointing.
			pad_index: ``int``: Padding token index for loss masking.
			attention_type: ``str``: Attention variant name (module under ``attention/``).
			output_attention: ``bool``: Whether to capture attention weights via hooks.
			tokenizer: ``BaseTokenizer | None``: Tokenizer for decoding in epoch metrics.
			optimizer: ``dict[str, Any]``: Optimizer config with ``cls`` key and kwargs.
			metrics: ``dict[str, list[BaseMetric]] | None``: Stage-keyed metrics.
		'''
		super().__init__(optimizer=optimizer, metrics=metrics)
		self.save_hyperparameters(ignore=['tokenizer', 'metrics'])
		self.max_len = max_len
		self.tokenizer = tokenizer
		self.criterion = nn.CrossEntropyLoss(ignore_index=pad_index)

		# model components
		self.src_embeddings = PosNTokEmbedding(src_vocab_size, emb_dim, max_len)
		self.tgt_embeddings = PosNTokEmbedding(tgt_vocab_size, emb_dim, max_len)
		self.encoder = TransformerEncoder(n_blocks, n_heads, emb_dim, dropout, bias,
										  use_grad_ckpt, attention_type, output_attention)
		self.decoder = TransformerDecoder(n_blocks, n_heads, emb_dim, dropout, bias,
										  use_grad_ckpt, attention_type, output_attention)
		self.lm_head = TransformerLMHead(emb_dim, tgt_vocab_size)

		# weight tying
		if weight_tying:
			if weight_tying == '3-way':
				assert src_vocab_size == tgt_vocab_size, \
					'3-way weight tying requires equal src and tgt vocab sizes.'
				self.src_embeddings.token_embedding_table.weight = self.lm_head.logits_head.weight
			self.tgt_embeddings.token_embedding_table.weight = self.lm_head.logits_head.weight

		# attention hooking
		if output_attention:
			self.attention_weights: dict[str, Tensor] = {}
			self._hook_attention_layers()

	def _hook_attention_layers(self) -> None:
		''' Register forward hooks on attention modules to capture their weights. '''
		def hook_fn(module, _input, output):
			self.attention_weights[module.tag] = output[1]
		for module in self.modules():
			if hasattr(module, 'tag') and hasattr(module, 'output_attention') and module.output_attention:
				module.register_forward_hook(hook_fn)

	# --- Forward methods ---

	def encoder_forward(self, src: Tensor, src_tok_mask: Tensor) -> Tensor:
		''' Forward pass through embeddings + encoder.

		Args:
			src: ``Tensor[(B, Ts), int64]``: Source token ids.
			src_tok_mask: ``Tensor[(B, Ts), bool]``: Source token mask.

		Returns:
			``Tensor[(B, Ts, D), float32]``: Encoder output.
		'''
		return self.encoder(self.src_embeddings(src), src_tok_mask)

	def incremental_forward(self, enc: Tensor, tgt: Tensor,
							src_tok_mask: Tensor, tgt_tok_mask: Tensor) -> Tensor:
		''' Forward through decoder + LM head for incremental decoding.

		Args:
			enc: ``Tensor[(B, Ts, D), float32]``: Encoder output.
			tgt: ``Tensor[(B, Tt), int64]``: Target token ids.
			src_tok_mask: ``Tensor[(B, Ts), bool]``: Source mask.
			tgt_tok_mask: ``Tensor[(B, Tt), bool]``: Target mask.

		Returns:
			``Tensor[(B, Tt, V), float32]``: Vocabulary logits.
		'''
		dec = self.decoder(enc, self.tgt_embeddings(tgt), src_tok_mask, tgt_tok_mask)
		return self.lm_head(dec)

	def forward(self, src: Tensor, tgt: Tensor,
				src_tok_mask: Tensor, tgt_tok_mask: Tensor) -> Tensor:
		''' Full forward pass: embed, encode, decode, project to logits.

		Args:
			src: ``Tensor[(B, Ts), int64]``: Source token ids.
			tgt: ``Tensor[(B, Tt), int64]``: Target token ids.
			src_tok_mask: ``Tensor[(B, Ts), bool]``: Source token mask.
			tgt_tok_mask: ``Tensor[(B, Tt), bool]``: Target token mask.

		Returns:
			``Tensor[(B, Tt, V), float32]``: Vocabulary logits.
		'''
		enc = self.encoder(self.src_embeddings(src), src_tok_mask)
		dec = self.decoder(enc, self.tgt_embeddings(tgt), src_tok_mask, tgt_tok_mask)
		return self.lm_head(dec)

	# --- BaseModel contract ---

	def produce(self, batch: dict[str, Any], requested: set[str]) -> dict[str, Any]:
		''' Produce requested outputs from a batch.

		Supports ``decoded_greedy`` and ``decoded_beam`` which trigger autoregressive
		decoding per sample (expensive, intended for epoch metrics).

		Args:
			batch: ``dict[str, Any]``: Input batch with keys ``x_src``, ``x_tgt``,
				``x_src_mask``, ``x_tgt_mask``, ``y_tgt``.
			requested: ``set[str]``: Subset of ``supports()``.

		Returns:
			``dict[str, Any]``: Dict of requested outputs.
		'''
		results: dict[str, Any] = {}

		# teacher-forced forward pass (needed for loss and y_pred)
		needs_forward = requested & {'loss', 'y_pred'}
		if needs_forward:
			y_pred = self(batch['x_src'], batch['x_tgt'], batch['x_src_mask'], batch['x_tgt_mask'])
			if 'loss' in requested:
				B, T = batch['y_tgt'].shape
				results['loss'] = self.criterion(y_pred.view(B * T, -1), batch['y_tgt'].reshape(B * T))
			if 'y_pred' in requested:
				results['y_pred'] = y_pred

		if 'y_true' in requested:
			results['y_true'] = batch['y_tgt']

		# autoregressive decoding (expensive, for epoch metrics)
		if 'decoded_greedy' in requested:
			results['decoded_greedy'] = self._decode_batch(
				batch['x_src'], sampling='argmax', max_new_tokens=128)
		if 'decoded_beam' in requested:
			results['decoded_beam'] = self._decode_batch_beam(
				batch['x_src'], beam_width=16, max_new_tokens=128)

		return results

	def supports(self) -> set[str]:
		''' Return the set of all output keys this model can produce. '''
		return {'loss', 'y_pred', 'y_true', 'decoded_greedy', 'decoded_beam'}

	# --- Batch decoding helpers (used by produce) ---

	def _decode_batch(self, x_src: Tensor, sampling: str = 'argmax',
					  max_new_tokens: int = 128) -> list[list[int]]:
		''' Autoregressively decode each sample in a batch using sampling.

		Args:
			x_src: ``Tensor[(B, Ts), int64]``: Source token ids.
			sampling: ``str``: ``'argmax'`` or ``'multinomial'``.
			max_new_tokens: ``int``: Maximum tokens to generate per sample.

		Returns:
			``list[list[int]]``: Decoded token lists, one per sample.
		'''
		decoded = []
		for i in range(x_src.size(0)):
			tokens = list(self.translate_with_sampling(
				x_src[i], self.BOS_IDX, self.EOS_IDX, sampling=sampling, max_new_tokens=max_new_tokens))
			decoded.append(tokens)
		return decoded

	def _decode_batch_beam(self, x_src: Tensor, beam_width: int = 16,
						   max_new_tokens: int = 128) -> list[list[int]]:
		''' Autoregressively decode each sample using beam search (top beam).

		Args:
			x_src: ``Tensor[(B, Ts), int64]``: Source token ids.
			beam_width: ``int``: Number of beams.
			max_new_tokens: ``int``: Maximum tokens to generate per sample.

		Returns:
			``list[list[int]]``: Decoded token lists (top beam), one per sample.
		'''
		decoded = []
		for i in range(x_src.size(0)):
			tokens = [int(beams[0]) for beams in self.translate_with_beams(
				x_src[i], self.BOS_IDX, self.EOS_IDX, beam_width=beam_width, max_new_tokens=max_new_tokens)]
			decoded.append(tokens)
		return decoded

	# --- Translation methods ---

	@torch.inference_mode()
	def translate_with_sampling(self, src: Tensor, bos_idx: int, eos_idx: int,
								sampling: str = 'multinomial', temperature: float = 1.0,
								max_new_tokens: int = 1000) -> Generator[int, None, None]:
		''' Translate a source sequence token-by-token using sampling or argmax.

		Args:
			src: ``Tensor[(Ts,), int64]``: Source token ids (unbatched).
			bos_idx: ``int``: Beginning-of-sequence token id.
			eos_idx: ``int``: End-of-sequence token id.
			sampling: ``str``: ``'multinomial'`` or ``'argmax'``.
			temperature: ``float``: Softmax temperature.
			max_new_tokens: ``int``: Maximum tokens to generate.

		Yields:
			``int``: Next generated token id.
		'''
		assert sampling in ['multinomial', 'argmax'], f'Invalid sampling: {sampling}'
		self.eval()

		src = src.to(self.device).unsqueeze(0)  # (1, Ts)
		src_mask = torch.ones_like(src, dtype=torch.bool, device=self.device)
		enc = self.encoder_forward(src[:, -self.max_len:], src_mask)  # (1, Ts, D)
		tgt = torch.tensor([bos_idx], dtype=torch.long, device=self.device).unsqueeze(0)  # (1, 1)

		for _ in range(max_new_tokens):
			tgt_mask = torch.ones_like(tgt, dtype=torch.bool, device=self.device)
			logits = self.incremental_forward(enc, tgt[:, -self.max_len:], src_mask, tgt_mask)
			logits = logits[:, -1, :] / temperature  # (1, V)
			probs = F.softmax(logits, dim=-1)
			if sampling == 'multinomial':
				idx_next = torch.multinomial(probs, num_samples=1)
			else:
				idx_next = torch.argmax(probs, dim=-1, keepdim=True)
			tgt = torch.cat((tgt, idx_next), dim=1)
			token = idx_next.item()
			yield token
			if token == eos_idx:
				break

	@torch.inference_mode()
	def translate_with_beams(self, src: Tensor, bos_idx: int, eos_idx: int,
							 beam_width: int = 16, max_new_tokens: int = 1000) -> Generator[np.ndarray, None, None]:
		''' Translate a source sequence using beam search.

		Args:
			src: ``Tensor[(Ts,), int64]``: Source token ids (unbatched).
			bos_idx: ``int``: Beginning-of-sequence token id.
			eos_idx: ``int``: End-of-sequence token id.
			beam_width: ``int``: Number of beams to maintain.
			max_new_tokens: ``int``: Maximum tokens to generate.

		Yields:
			``np.ndarray[(beam_width,), int64]``: Next token for each beam.
		'''
		self.eval()

		src = src.to(self.device).unsqueeze(0)  # (1, Ts)
		src_mask = torch.ones_like(src, dtype=torch.bool, device=self.device)
		enc = self.encoder_forward(src[:, -self.max_len:], src_mask)  # (1, Ts, D)
		tgt = torch.tensor([[bos_idx]], dtype=torch.long, device=self.device)  # (1, 1)
		tgt_mask = torch.ones_like(tgt, dtype=torch.bool, device=self.device)
		tgt_probs = torch.ones(beam_width, dtype=torch.float, device=self.device)
		eos_reached = torch.zeros(beam_width, dtype=torch.bool, device=self.device)

		# Stage 1: first expansion
		logits = self.incremental_forward(enc, tgt, src_mask, tgt_mask)
		logits = logits[0, 0, :]  # (V,)
		_, topk_idx = torch.topk(logits, k=beam_width, dim=-1)
		eos_reached = eos_reached | (topk_idx == eos_idx)
		yield topk_idx.cpu().numpy()

		# Stage 1+T: expand beams
		enc = enc.repeat(beam_width, 1, 1)  # (Bw, Ts, D)
		src_mask = src_mask.repeat(beam_width, 1)  # (Bw, Ts)
		tgt = torch.concat((tgt.repeat(beam_width, 1), topk_idx.unsqueeze(1)), dim=1)  # (Bw, 2)

		for _ in range(max_new_tokens - 1):
			if torch.all(eos_reached):
				break
			tgt = tgt[:, -self.max_len:]
			tgt_mask = torch.ones_like(tgt, dtype=torch.bool, device=self.device)
			logits = self.incremental_forward(enc, tgt, src_mask, tgt_mask)
			logits = logits[:, -1, :]  # (Bw, V)
			next_probs = F.softmax(logits, dim=-1)
			joint_probs = tgt_probs.unsqueeze(1) * next_probs  # (Bw, V)
			topk_probs, topk_flat = torch.topk(joint_probs.flatten(), k=beam_width, dim=-1)
			# decompose flat indices into (beam_idx, vocab_idx)
			V = joint_probs.size(1)
			parent_beams = topk_flat // V  # (Bw,)
			next_tokens = topk_flat % V  # (Bw,)
			eos_reached = eos_reached[parent_beams] | (next_tokens == eos_idx)
			# reorder beams by parent lineage and append new tokens
			tgt = torch.concat((tgt[parent_beams], next_tokens.unsqueeze(1)), dim=1)  # (Bw, T+1)
			tgt_probs = topk_probs
			yield next_tokens.cpu().numpy()
			if torch.all(eos_reached):
				break
