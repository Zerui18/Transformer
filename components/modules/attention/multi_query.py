import math

import torch
from torch import nn, Tensor
from einops import rearrange

from components.modules.attention.base import MultiHeadSelfAttentionBase, MultiHeadCrossAttentionBase


class MultiQuerySelfAttention(MultiHeadSelfAttentionBase):
	'''Multi-query self-attention using combined QKV projection and einsum-based mask construction.

	Properties:
		1. attn_dropout: nn.Dropout  dropout applied to attention weights.
		2. resid_dropout: nn.Dropout  dropout applied after the output projection.
		3. qkv_projection: nn.Linear  combined Q, K, V linear projection (D -> 3D).
		4. c_proj: nn.Linear  output linear projection (D -> D).

	Identical structure to VanillaSelfAttention but uses torch.einsum for constructing
	the token mask from per-position masks. Uses einops for head reshaping.
	'''

	def __init__(self, *args, **kwargs) -> None:
		'''Initialise MultiQuerySelfAttention layers.

		Args:
			1. *args: passed to MultiHeadSelfAttentionBase.
			2. **kwargs: passed to MultiHeadSelfAttentionBase.
		'''
		super().__init__(*args, **kwargs)
		self.attn_dropout = nn.Dropout(self.dropout)
		self.resid_dropout = nn.Dropout(self.dropout)
		# combine q, k, v projections for efficiency
		self.qkv_projection = nn.Linear(self.emb_dim, 3 * self.emb_dim, bias=self.bias)
		# output projection
		self.c_proj = nn.Linear(self.emb_dim, self.emb_dim, bias=self.bias)

	def _forward(self, x: Tensor, tok_mask: Tensor) -> tuple[Tensor, Tensor]:
		'''Compute self-attention via manual scaled dot-product with einsum mask construction.

		Args:
			1. x: Tensor  [float32, (B, T, D)] input embeddings.
			2. tok_mask: Tensor  [bool, (B, T)] per-token mask; False is masked out, True is preserved.
		Returns:
			result: tuple[Tensor, Tensor]  (output (B, T, D), attention_weights (B, H, T, T)).
		'''
		B, T, D = x.shape
		H = self.n_heads
		# proj q, k, v for all heads
		q, k, v = self.qkv_projection(x).split(self.emb_dim, dim=2)
		q = rearrange(q, 'B T (H Dh) -> B H T Dh', H=H)  # (B, H, T, D//H)
		k = rearrange(k, 'B T (H Dh) -> B H T Dh', H=H)  # (B, H, T, D//H)
		v = rearrange(v, 'B T (H Dh) -> B H T Dh', H=H)  # (B, H, T, D//H)
		# compute attention scores
		att_weights = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))  # (B, H, T, T)
		# construct mask via einsum outer product of token masks
		mask = torch.einsum('bt,bT->btT', tok_mask.float(), tok_mask.float()) > 0  # (B, T, T)
		mask = mask.unsqueeze(1)  # (B, 1, T, T)
		if self.is_causal:
			causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
			mask = mask & causal_mask[None, None, :, :]
		att_weights = att_weights.masked_fill(mask == 0, -1e9)
		att_weights = nn.functional.softmax(att_weights, dim=-1)
		y = self.attn_dropout(att_weights) @ v  # (B, H, T, D//H)
		# combine heads
		y = rearrange(y, 'B H T Dh -> B T (H Dh)')  # (B, T, D)
		y = self.resid_dropout(self.c_proj(y))
		return y, att_weights


class MultiQueryCrossAttention(MultiHeadCrossAttentionBase):
	'''Multi-query cross-attention using separate Q and combined KV projections with einsum-based mask construction.

	Properties:
		1. attn_dropout: nn.Dropout  dropout applied to attention weights.
		2. resid_dropout: nn.Dropout  dropout applied after the output projection.
		3. q_projection: nn.Linear  query linear projection (D -> D).
		4. kv_projection: nn.Linear  combined K, V linear projection (D -> 2D).
		5. c_proj: nn.Linear  output linear projection (D -> D).

	Identical structure to VanillaCrossAttention but uses torch.einsum for constructing
	the cross-attention mask. Uses einops for head reshaping.
	'''

	def __init__(self, *args, **kwargs) -> None:
		'''Initialise MultiQueryCrossAttention layers.

		Args:
			1. *args: passed to MultiHeadCrossAttentionBase.
			2. **kwargs: passed to MultiHeadCrossAttentionBase.
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
		'''Compute cross-attention via manual scaled dot-product with einsum mask construction.

		Args:
			1. x_q: Tensor  [float32, (B, Tq, D)] query embeddings.
			2. x_kv: Tensor  [float32, (B, Tk, D)] key/value embeddings.
			3. q_tok_mask: Tensor  [bool, (B, Tq)] query token mask; False is masked out.
			4. kv_tok_mask: Tensor  [bool, (B, Tk)] key/value token mask; False is masked out.
		Returns:
			result: tuple[Tensor, Tensor]  (output (B, Tq, D), attention_weights (B, H, Tq, Tk)).
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
		# compute attention scores
		att_weights = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))  # (B, H, Tq, Tk)
		# construct mask via einsum outer product of token masks
		attn_mask = torch.einsum('bt,bT->btT', q_tok_mask.float(), kv_tok_mask.float()) > 0  # (B, Tq, Tk)
		# apply mask
		att_weights = att_weights.masked_fill(attn_mask.unsqueeze(1) == 0, -1e9)
		att_weights = nn.functional.softmax(att_weights, dim=-1)
		y = self.attn_dropout(att_weights) @ v  # (B, H, Tq, D//H)
		# combine heads
		y = rearrange(y, 'B H Tq Dh -> B Tq (H Dh)')  # (B, Tq, D)
		y = self.resid_dropout(self.c_proj(y))
		return y, att_weights

# Module-level aliases for lookup by transformer blocks
SELF_ATTENTION_CLS = MultiQuerySelfAttention
CROSS_ATTENTION_CLS = MultiQueryCrossAttention
