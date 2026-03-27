import math
import functools

import torch
from torch import nn, Tensor
from einops import rearrange

from components.modules.attention.base import MultiHeadSelfAttentionBase, MultiHeadCrossAttentionBase


### RotaryEmbedding Helper Functions ###

def get_angles(theta: float, seq_len: int, hidden_dim: int) -> Tensor:
	'''Compute rotary embedding angle matrix for the given sequence length and hidden dimension.

	Args:
		1. theta: float  base frequency for the rotary embedding.
		2. seq_len: int  maximum sequence length.
		3. hidden_dim: int  per-head hidden dimension (must be even).
	Returns:
		angles: Tensor  [float32, (seq_len, hidden_dim // 2)] angle matrix.
	'''
	# angular speed
	w = 1.0 / (theta ** (torch.arange(1, hidden_dim + 1, 2, dtype=torch.float)[:(hidden_dim // 2)] / hidden_dim))
	# time
	t = torch.arange(1, seq_len + 1, dtype=torch.float)
	angles = torch.einsum('i, j -> i j', t, w)
	return angles


def get_sin_cos(angles: Tensor) -> tuple[Tensor, Tensor]:
	'''Compute sin and cos from an angle matrix.

	Args:
		1. angles: Tensor  [float32, (T, hidden_dim // 2)] angle matrix.
	Returns:
		sin_cos: tuple[Tensor, Tensor]  (sin (T, hidden_dim // 2), cos (T, hidden_dim // 2)).
	'''
	return torch.sin(angles), torch.cos(angles)


def rotate_len2_subvectors(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
	'''Treat the last dimension as pairs of values and rotate each pair by the given angles.

	Args:
		1. x: Tensor  [float32, (..., T, Dh)] input tensor where Dh is even.
		2. sin: Tensor  [float32, (T, Dh // 2)] sin values for rotation.
		3. cos: Tensor  [float32, (T, Dh // 2)] cos values for rotation.
	Returns:
		rotated: Tensor  [float32, (..., T, Dh)] rotated tensor, same shape as x.

	Rotation formula for each pair (x1, x2):
		x1' = x1 * cos - x2 * sin
		x2' = x1 * sin + x2 * cos
	'''
	assert x.shape[-1] % 2 == 0, 'x.shape[-1] must be even'
	assert sin.shape[-1] == cos.shape[-1] == x.shape[-1] // 2, \
		'sin.shape[-1] must equal cos.shape[-1] and x.shape[-1] // 2'
	x1 = x[..., ::2]
	x2 = x[..., 1::2]
	x1_prime = x1 * cos - x2 * sin
	x2_prime = x1 * sin + x2 * cos
	return torch.stack((x1_prime, x2_prime), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
	'''Rotary positional embedding (RoPE) from https://arxiv.org/abs/2104.09864.

	Properties:
		1. theta: float  base frequency parameter.
		2. hidden_dim: int  per-head hidden dimension.

	Applies rotary positional embeddings to query or key tensors in an attention module.
	Sin/cos tables are cached at the class level for efficiency and computed up to a
	hardcoded maximum sequence length of 8192.

	Validated against lucidrains/rotary-embedding-torch with max diff < 1e-6.
	'''

	@staticmethod
	@functools.cache
	def get_sin_cos(theta: float, hidden_dim: int, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
		'''Get cached sin/cos tables for the given theta and hidden_dim.

		Args:
			1. theta: float  base frequency parameter.
			2. hidden_dim: int  per-head hidden dimension.
			3. dtype: torch.dtype  desired output dtype.
		Returns:
			sin_cos: tuple[Tensor, Tensor]  (sin (8192, hidden_dim // 2), cos (8192, hidden_dim // 2)).
		'''
		MAX_SEQ_LEN = 8192  # hardcoded to avoid recomputing the sin/cos tables
		angles = get_angles(theta, MAX_SEQ_LEN, hidden_dim)
		sin, cos = get_sin_cos(angles)
		return sin.type(dtype), cos.type(dtype)

	def __init__(self, theta: float, hidden_dim: int) -> None:
		'''Initialise RotaryEmbedding.

		Args:
			1. theta: float  base frequency parameter (typically 10000).
			2. hidden_dim: int  per-head hidden dimension (must be even).
		'''
		super().__init__()
		self.theta = theta
		self.hidden_dim = hidden_dim

	def forward(self, x: Tensor) -> Tensor:
		'''Apply rotary positional embedding to input tensor.

		Args:
			1. x: Tensor  [float32, (..., T, Dh)] query or key tensor to rotate.
		Returns:
			rotated: Tensor  [float32, (..., T, Dh)] rotated tensor, same shape as x.
		'''
		# get sequence length
		T = x.shape[-2]
		# get sin/cos tables
		sin, cos = RotaryEmbedding.get_sin_cos(self.theta, self.hidden_dim, x.dtype)
		# trim to the correct sequence length and move to device
		sin = sin[:T, :].to(x.device)
		cos = cos[:T, :].to(x.device)
		# rotate x
		return rotate_len2_subvectors(x, sin, cos)


### Attention Modules ###

class RoFormerSelfAttention(MultiHeadSelfAttentionBase):
	'''Multi-head self-attention with rotary positional embeddings (RoPE) applied to queries and keys.

	Properties:
		1. attn_dropout: nn.Dropout  dropout applied to attention weights.
		2. resid_dropout: nn.Dropout  dropout applied after the output projection.
		3. qkv_projection: nn.Linear  combined Q, K, V linear projection (D -> 3D).
		4. c_proj: nn.Linear  output linear projection (D -> D).
		5. rotary_embedding: RotaryEmbedding  rotary positional embedding applied to Q and K.

	RoPE encodes relative position information by rotating query and key vectors,
	enabling length generalisation without explicit positional encodings.
	'''

	def __init__(self, *args, **kwargs) -> None:
		'''Initialise RoFormerSelfAttention layers.

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
		self.rotary_embedding = RotaryEmbedding(theta=10000, hidden_dim=self.emb_dim // self.n_heads)

	def _forward(self, x: Tensor, tok_mask: Tensor) -> tuple[Tensor, Tensor]:
		'''Compute self-attention with rotary positional embeddings on Q and K.

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
		# apply rotary embedding to q, k
		q = self.rotary_embedding(q)
		k = self.rotary_embedding(k)
		# compute attention scores
		att_weights = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))  # (B, H, T, T)
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
		y = self.attn_dropout(att_weights) @ v  # (B, H, T, D//H)
		# combine heads
		y = rearrange(y, 'B H T Dh -> B T (H Dh)')  # (B, T, D)
		y = self.resid_dropout(self.c_proj(y))
		return y, att_weights


class RoFormerCrossAttention(MultiHeadCrossAttentionBase):
	'''Multi-head cross-attention with rotary positional embeddings (RoPE) applied to queries and keys.

	Properties:
		1. attn_dropout: nn.Dropout  dropout applied to attention weights.
		2. resid_dropout: nn.Dropout  dropout applied after the output projection.
		3. q_projection: nn.Linear  query linear projection (D -> D).
		4. kv_projection: nn.Linear  combined K, V linear projection (D -> 2D).
		5. c_proj: nn.Linear  output linear projection (D -> D).
		6. rotary_embedding: RotaryEmbedding  rotary positional embedding applied to Q and K.

	RoPE encodes relative position information by rotating query and key vectors,
	enabling length generalisation without explicit positional encodings.
	'''

	def __init__(self, *args, **kwargs) -> None:
		'''Initialise RoFormerCrossAttention layers.

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
		self.rotary_embedding = RotaryEmbedding(theta=10000, hidden_dim=self.emb_dim // self.n_heads)

	def _forward(self, x_q: Tensor, x_kv: Tensor, q_tok_mask: Tensor, kv_tok_mask: Tensor) -> tuple[Tensor, Tensor]:
		'''Compute cross-attention with rotary positional embeddings on Q and K.

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
		# apply rotary embedding to q, k
		q = self.rotary_embedding(q)
		k = self.rotary_embedding(k)
		# compute attention scores
		att_weights = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))  # (B, H, Tq, Tk)
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
SELF_ATTENTION_CLS = RoFormerSelfAttention
CROSS_ATTENTION_CLS = RoFormerCrossAttention
