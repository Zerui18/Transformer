import math

import torch
from torch import nn, Tensor
from einops import rearrange

from components.modules.attention.base import MultiHeadSelfAttentionBase, MultiHeadCrossAttentionBase


class AverageSelfAttention(MultiHeadSelfAttentionBase):
	'''Self-attention with uniform (average) attention weights.

	Instead of computing Q@K^T scores, uses uniform weights across all unmasked
	positions. Only projects V since Q and K are unused.

	Attributes:
		attn_dropout: ``nn.Dropout``: dropout applied to attention weights.
		resid_dropout: ``nn.Dropout``: dropout applied after the output projection.
		v_projection: ``nn.Linear``: value projection (D -> D).
		c_proj: ``nn.Linear``: output linear projection (D -> D).
	'''

	def __init__(self, *args, **kwargs) -> None:
		'''Initialise AverageSelfAttention layers.

		Args:
			*args: passed to MultiHeadSelfAttentionBase.
			**kwargs: passed to MultiHeadSelfAttentionBase.
		'''
		super().__init__(*args, **kwargs)
		self.attn_dropout = nn.Dropout(self.dropout)
		self.resid_dropout = nn.Dropout(self.dropout)
		self.v_projection = nn.Linear(self.emb_dim, self.emb_dim, bias=self.bias)
		self.c_proj = nn.Linear(self.emb_dim, self.emb_dim, bias=self.bias)

	def _forward(self, x: Tensor, tok_mask: Tensor) -> tuple[Tensor, Tensor]:
		'''Compute self-attention with uniform attention weights.

		Args:
			x: ``Tensor[(B, T, D), float32]``: input embeddings.
			tok_mask: ``Tensor[(B, T), bool]``: per-token mask; False is masked out.

		Returns:
			``tuple[Tensor, Tensor]``: (output (B, T, D), attention_weights (B, H, T, T)).
		'''
		B, T, D = x.shape
		H = self.n_heads
		Dh = D // H
		v = rearrange(self.v_projection(x), 'B T (H Dh) -> B H T Dh', H=H)  # (B, H, T, Dh)
		# uniform attention weights
		att_weights = torch.ones((B, H, T, T), dtype=x.dtype, device=x.device) / math.sqrt(Dh)
		# token mask
		mask = tok_mask[:, None, :] & tok_mask[:, :, None]  # (B, T, T)
		if self.is_causal:
			causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
			mask = mask & causal
		att_weights = att_weights.masked_fill(mask.unsqueeze(1) == 0, -1e9)
		att_weights = nn.functional.softmax(att_weights, dim=-1)
		att_weights = self.attn_dropout(att_weights)
		y = att_weights @ v  # (B, H, T, Dh)
		y = rearrange(y, 'B H T Dh -> B T (H Dh)')  # (B, T, D)
		y = self.resid_dropout(self.c_proj(y))
		return y, att_weights


class AverageCrossAttention(MultiHeadCrossAttentionBase):
	'''Cross-attention with uniform (average) attention weights.

	Instead of computing Q@K^T scores, uses uniform weights across all unmasked
	positions. Only projects V since Q and K are unused.

	Attributes:
		attn_dropout: ``nn.Dropout``: dropout applied to attention weights.
		resid_dropout: ``nn.Dropout``: dropout applied after the output projection.
		v_projection: ``nn.Linear``: value projection (D -> D).
		c_proj: ``nn.Linear``: output linear projection (D -> D).
	'''

	def __init__(self, *args, **kwargs) -> None:
		'''Initialise AverageCrossAttention layers.

		Args:
			*args: passed to MultiHeadCrossAttentionBase.
			**kwargs: passed to MultiHeadCrossAttentionBase.
		'''
		super().__init__(*args, **kwargs)
		self.attn_dropout = nn.Dropout(self.dropout)
		self.resid_dropout = nn.Dropout(self.dropout)
		self.v_projection = nn.Linear(self.emb_dim, self.emb_dim, bias=self.bias)
		self.c_proj = nn.Linear(self.emb_dim, self.emb_dim, bias=self.bias)

	def _forward(self, x_q: Tensor, x_kv: Tensor, q_tok_mask: Tensor, kv_tok_mask: Tensor) -> tuple[Tensor, Tensor]:
		'''Compute cross-attention with uniform attention weights.

		Args:
			x_q: ``Tensor[(B, Tq, D), float32]``: query embeddings (used only for shape).
			x_kv: ``Tensor[(B, Tk, D), float32]``: key/value embeddings.
			q_tok_mask: ``Tensor[(B, Tq), bool]``: query token mask.
			kv_tok_mask: ``Tensor[(B, Tk), bool]``: key/value token mask.

		Returns:
			``tuple[Tensor, Tensor]``: (output (B, Tq, D), attention_weights (B, H, Tq, Tk)).
		'''
		B, Tq, D = x_q.shape
		_, Tk, _ = x_kv.shape
		H = self.n_heads
		Dh = D // H
		v = rearrange(self.v_projection(x_kv), 'B Tk (H Dh) -> B H Tk Dh', H=H)  # (B, H, Tk, Dh)
		# uniform attention weights
		att_weights = torch.ones((B, H, Tq, Tk), dtype=x_q.dtype, device=x_q.device) / math.sqrt(Dh)
		# token mask
		mask = q_tok_mask[:, :, None] & kv_tok_mask[:, None, :]  # (B, Tq, Tk)
		att_weights = att_weights.masked_fill(mask.unsqueeze(1) == 0, -1e9)
		att_weights = nn.functional.softmax(att_weights, dim=-1)
		att_weights = self.attn_dropout(att_weights)
		y = att_weights @ v  # (B, H, Tq, Dh)
		y = rearrange(y, 'B H Tq Dh -> B Tq (H Dh)')  # (B, Tq, D)
		y = self.resid_dropout(self.c_proj(y))
		return y, att_weights

# Module-level aliases for lookup by transformer blocks
SELF_ATTENTION_CLS = AverageSelfAttention
CROSS_ATTENTION_CLS = AverageCrossAttention
