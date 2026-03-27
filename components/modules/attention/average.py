import math

import torch
from torch import nn, Tensor
from einops import rearrange

from components.modules.attention.base import MultiHeadSelfAttentionBase, MultiHeadCrossAttentionBase


class AverageSelfAttention(MultiHeadSelfAttentionBase):
	'''Multi-head self-attention using uniform (average) attention weights instead of learned Q@K^T scores.

	Attributes:
		attn_dropout: ``nn.Dropout``: dropout applied to attention weights.
		resid_dropout: ``nn.Dropout``: dropout applied after the output projection.
		qkv_projection: ``nn.Linear``: combined Q, K, V linear projection (D -> 3D).
		c_proj: ``nn.Linear``: output linear projection (D -> D).

	Attention weights are initialised to all ones / sqrt(dk) instead of being computed
	from query-key dot products. Masking and softmax are still applied, so unmasked
	positions receive uniform attention after normalisation.
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
		# combine q, k, v projections for efficiency
		self.qkv_projection = nn.Linear(self.emb_dim, 3 * self.emb_dim, bias=self.bias)
		# output projection
		self.c_proj = nn.Linear(self.emb_dim, self.emb_dim, bias=self.bias)

	def _forward(self, x: Tensor, tok_mask: Tensor) -> tuple[Tensor, Tensor]:
		'''Compute self-attention with uniform attention weights.

		Args:
			x: ``Tensor[(B, T, D), float32]``: input embeddings.
			tok_mask: ``Tensor[(B, T), bool]``: per-token mask; False is masked out, True is preserved.
		Returns:
			``tuple[Tensor, Tensor]``: (output (B, T, D), attention_weights (B, H, T, T)).
		'''
		B, T, D = x.shape
		H = self.n_heads
		# proj q, k, v for all heads (v is used; q, k only for dimension consistency)
		q, k, v = self.qkv_projection(x).split(self.emb_dim, dim=2)
		q = rearrange(q, 'B T (H Dh) -> B H T Dh', H=H)  # (B, H, T, D//H)
		k = rearrange(k, 'B T (H Dh) -> B H T Dh', H=H)  # (B, H, T, D//H)
		v = rearrange(v, 'B T (H Dh) -> B H T Dh', H=H)  # (B, H, T, D//H)
		# uniform attention weights: all ones scaled by 1/sqrt(dk)
		att_weights = torch.ones((B, H, T, T), dtype=torch.float, device=x.device) / math.sqrt(k.size(-1))
		# construct token mask
		mask = tok_mask.view(B, 1, T)   # (B, 1, T)
		mask = mask.tile(1, T, 1)       # (B, T, T)
		mask = mask & mask.transpose(-2, -1)  # (B, T, T)
		mask = mask.view(B, 1, T, T)    # (B, 1, T, T)
		if self.is_causal:
			causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
			mask = mask & causal_mask[None, None, :, :]
		att_weights = att_weights.masked_fill(mask == 0, -1e9)
		att_weights = nn.functional.softmax(att_weights, dim=-1)
		att_weights = self.attn_dropout(att_weights)
		y = att_weights @ v  # (B, H, T, D//H)
		# combine heads
		y = rearrange(y, 'B H T Dh -> B T (H Dh)')  # (B, T, D)
		y = self.resid_dropout(self.c_proj(y))
		return y, att_weights


class AverageCrossAttention(MultiHeadCrossAttentionBase):
	'''Multi-head cross-attention using uniform (average) attention weights instead of learned Q@K^T scores.

	Attributes:
		attn_dropout: ``nn.Dropout``: dropout applied to attention weights.
		resid_dropout: ``nn.Dropout``: dropout applied after the output projection.
		q_projection: ``nn.Linear``: query linear projection (D -> D).
		kv_projection: ``nn.Linear``: combined K, V linear projection (D -> 2D).
		c_proj: ``nn.Linear``: output linear projection (D -> D).

	Attention weights are initialised to all ones / sqrt(dk) instead of being computed
	from query-key dot products. Masking and softmax are still applied, so unmasked
	positions receive uniform attention after normalisation.
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
		self.q_projection = nn.Linear(self.emb_dim, self.emb_dim, bias=self.bias)
		# combine k, v projections for efficiency
		self.kv_projection = nn.Linear(self.emb_dim, 2 * self.emb_dim, bias=self.bias)
		# output projection
		self.c_proj = nn.Linear(self.emb_dim, self.emb_dim, bias=self.bias)

	def _forward(self, x_q: Tensor, x_kv: Tensor, q_tok_mask: Tensor, kv_tok_mask: Tensor) -> tuple[Tensor, Tensor]:
		'''Compute cross-attention with uniform attention weights.

		Args:
			x_q: ``Tensor[(B, Tq, D), float32]``: query embeddings.
			x_kv: ``Tensor[(B, Tk, D), float32]``: key/value embeddings.
			q_tok_mask: ``Tensor[(B, Tq), bool]``: query token mask; False is masked out.
			kv_tok_mask: ``Tensor[(B, Tk), bool]``: key/value token mask; False is masked out.
		Returns:
			``tuple[Tensor, Tensor]``: (output (B, Tq, D), attention_weights (B, H, Tq, Tk)).
		'''
		B, Tq, D = x_q.shape
		_, Tk, _ = x_kv.shape
		H = self.n_heads
		# proj query for all heads
		q = self.q_projection(x_q)
		q = rearrange(q, 'B Tq (H Dh) -> B H Tq Dh', H=H)  # (B, H, Tq, D//H)
		# proj key & value for all heads
		k, v = self.kv_projection(x_kv).split(self.emb_dim, dim=2)
		k = rearrange(k, 'B Tk (H Dh) -> B H Tk Dh', H=H)  # (B, H, Tk, D//H)
		v = rearrange(v, 'B Tk (H Dh) -> B H Tk Dh', H=H)  # (B, H, Tk, D//H)
		# uniform attention weights: all ones scaled by 1/sqrt(dk)
		att_weights = torch.ones((B, H, Tq, Tk), dtype=torch.float, device=x_q.device) / math.sqrt(k.size(-1))
		# merge masks
		q_mask = q_tok_mask.unsqueeze(2)   # (B, Tq, 1)
		kv_mask = kv_tok_mask.unsqueeze(1)  # (B, 1, Tk)
		attn_mask = q_mask & kv_mask        # (B, Tq, Tk)
		# apply mask
		att_weights = att_weights.masked_fill(attn_mask.unsqueeze(1) == 0, -1e9)
		att_weights = nn.functional.softmax(att_weights, dim=-1)
		att_weights = self.attn_dropout(att_weights)
		y = att_weights @ v  # (B, H, Tq, D//H)
		# combine heads
		y = rearrange(y, 'B H Tq Dh -> B Tq (H Dh)')  # (B, Tq, D)
		y = self.resid_dropout(self.c_proj(y))
		return y, att_weights

# Module-level aliases for lookup by transformer blocks
SELF_ATTENTION_CLS = AverageSelfAttention
CROSS_ATTENTION_CLS = AverageCrossAttention
