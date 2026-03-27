import math

import torch
from torch import nn, Tensor
from einops import rearrange

from components.modules.attention.base import MultiHeadSelfAttentionBase, MultiHeadCrossAttentionBase


class MultiQuerySelfAttention(MultiHeadSelfAttentionBase):
	'''Multi-Query self-attention (Shazeer 2019).

	Q has n_heads heads, K and V share a single head that is broadcast across
	all query heads. This reduces KV memory and computation by a factor of H
	while preserving most of the representational capacity.

	Attributes:
		attn_dropout: ``nn.Dropout``: dropout applied to attention weights.
		resid_dropout: ``nn.Dropout``: dropout applied after the output projection.
		q_projection: ``nn.Linear``: query projection (D -> D), split into H heads.
		kv_projection: ``nn.Linear``: shared K/V projection (D -> 2 * Dh), single head.
		c_proj: ``nn.Linear``: output linear projection (D -> D).
	'''

	def __init__(self, *args, **kwargs) -> None:
		'''Initialise MultiQuerySelfAttention layers.

		Args:
			*args: passed to MultiHeadSelfAttentionBase.
			**kwargs: passed to MultiHeadSelfAttentionBase.
		'''
		super().__init__(*args, **kwargs)
		self.head_dim = self.emb_dim // self.n_heads
		self.attn_dropout = nn.Dropout(self.dropout)
		self.resid_dropout = nn.Dropout(self.dropout)
		# Q gets H heads, K/V share 1 head
		self.q_projection = nn.Linear(self.emb_dim, self.emb_dim, bias=self.bias)
		self.kv_projection = nn.Linear(self.emb_dim, 2 * self.head_dim, bias=self.bias)
		self.c_proj = nn.Linear(self.emb_dim, self.emb_dim, bias=self.bias)

	def _forward(self, x: Tensor, tok_mask: Tensor) -> tuple[Tensor, Tensor]:
		'''Compute multi-query self-attention.

		Args:
			x: ``Tensor[(B, T, D), float32]``: input embeddings.
			tok_mask: ``Tensor[(B, T), bool]``: per-token mask; False is masked out.

		Returns:
			``tuple[Tensor, Tensor]``: (output (B, T, D), attention_weights (B, H, T, T)).
		'''
		B, T, D = x.shape
		H = self.n_heads
		Dh = self.head_dim
		# Q: H heads
		q = rearrange(self.q_projection(x), 'B T (H Dh) -> B H T Dh', H=H)  # (B, H, T, Dh)
		# K, V: single shared head, broadcast over H
		kv = self.kv_projection(x)  # (B, T, 2*Dh)
		k, v = kv.split(Dh, dim=2)
		k = k.unsqueeze(1)  # (B, 1, T, Dh) — broadcasts over H
		v = v.unsqueeze(1)  # (B, 1, T, Dh)
		# attention scores
		att_weights = (q @ k.transpose(-2, -1)) / math.sqrt(Dh)  # (B, H, T, T)
		# token mask
		mask = tok_mask[:, None, :] & tok_mask[:, :, None]  # (B, T, T)
		if self.is_causal:
			causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
			mask = mask & causal
		att_weights = att_weights.masked_fill(mask.unsqueeze(1) == 0, -1e9)
		att_weights = nn.functional.softmax(att_weights, dim=-1)
		y = self.attn_dropout(att_weights) @ v  # (B, H, T, Dh)
		y = rearrange(y, 'B H T Dh -> B T (H Dh)')  # (B, T, D)
		y = self.resid_dropout(self.c_proj(y))
		return y, att_weights


class MultiQueryCrossAttention(MultiHeadCrossAttentionBase):
	'''Multi-Query cross-attention (Shazeer 2019).

	Q has n_heads heads, K and V share a single head that is broadcast across
	all query heads.

	Attributes:
		attn_dropout: ``nn.Dropout``: dropout applied to attention weights.
		resid_dropout: ``nn.Dropout``: dropout applied after the output projection.
		q_projection: ``nn.Linear``: query projection (D -> D), split into H heads.
		kv_projection: ``nn.Linear``: shared K/V projection (D -> 2 * Dh), single head.
		c_proj: ``nn.Linear``: output linear projection (D -> D).
	'''

	def __init__(self, *args, **kwargs) -> None:
		'''Initialise MultiQueryCrossAttention layers.

		Args:
			*args: passed to MultiHeadCrossAttentionBase.
			**kwargs: passed to MultiHeadCrossAttentionBase.
		'''
		super().__init__(*args, **kwargs)
		self.head_dim = self.emb_dim // self.n_heads
		self.attn_dropout = nn.Dropout(self.dropout)
		self.resid_dropout = nn.Dropout(self.dropout)
		self.q_projection = nn.Linear(self.emb_dim, self.emb_dim, bias=self.bias)
		self.kv_projection = nn.Linear(self.emb_dim, 2 * self.head_dim, bias=self.bias)
		self.c_proj = nn.Linear(self.emb_dim, self.emb_dim, bias=self.bias)

	def _forward(self, x_q: Tensor, x_kv: Tensor, q_tok_mask: Tensor, kv_tok_mask: Tensor) -> tuple[Tensor, Tensor]:
		'''Compute multi-query cross-attention.

		Args:
			x_q: ``Tensor[(B, Tq, D), float32]``: query embeddings.
			x_kv: ``Tensor[(B, Tk, D), float32]``: key/value embeddings.
			q_tok_mask: ``Tensor[(B, Tq), bool]``: query token mask.
			kv_tok_mask: ``Tensor[(B, Tk), bool]``: key/value token mask.

		Returns:
			``tuple[Tensor, Tensor]``: (output (B, Tq, D), attention_weights (B, H, Tq, Tk)).
		'''
		B, Tq, D = x_q.shape
		_, Tk, _ = x_kv.shape
		H = self.n_heads
		Dh = self.head_dim
		# Q: H heads
		q = rearrange(self.q_projection(x_q), 'B Tq (H Dh) -> B H Tq Dh', H=H)  # (B, H, Tq, Dh)
		# K, V: single shared head
		kv = self.kv_projection(x_kv)  # (B, Tk, 2*Dh)
		k, v = kv.split(Dh, dim=2)
		k = k.unsqueeze(1)  # (B, 1, Tk, Dh)
		v = v.unsqueeze(1)  # (B, 1, Tk, Dh)
		# attention scores
		att_weights = (q @ k.transpose(-2, -1)) / math.sqrt(Dh)  # (B, H, Tq, Tk)
		# token mask
		mask = q_tok_mask[:, :, None] & kv_tok_mask[:, None, :]  # (B, Tq, Tk)
		att_weights = att_weights.masked_fill(mask.unsqueeze(1) == 0, -1e9)
		att_weights = nn.functional.softmax(att_weights, dim=-1)
		y = self.attn_dropout(att_weights) @ v  # (B, H, Tq, Dh)
		y = rearrange(y, 'B H Tq Dh -> B Tq (H Dh)')  # (B, Tq, D)
		y = self.resid_dropout(self.c_proj(y))
		return y, att_weights

# Module-level aliases for lookup by transformer blocks
SELF_ATTENTION_CLS = MultiQuerySelfAttention
CROSS_ATTENTION_CLS = MultiQueryCrossAttention
