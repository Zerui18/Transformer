from torch import nn, Tensor
from einops import rearrange
from xformers.ops import memory_efficient_attention, LowerTriangularMask

from components.modules.attention.base import MultiHeadSelfAttentionBase, MultiHeadCrossAttentionBase


class XformerSelfAttention(MultiHeadSelfAttentionBase):
	'''Multi-head self-attention using xformers memory_efficient_attention backend.

	Properties:
		1. resid_dropout: nn.Dropout  dropout applied after the output projection.
		2. qkv_projection: nn.Linear  combined Q, K, V linear projection (D -> 3D).
		3. c_proj: nn.Linear  output linear projection (D -> D).

	Uses xformers.ops.memory_efficient_attention which expects tensors in (B, T, H, Dh)
	layout. Causal masking is applied via LowerTriangularMask. Attention weights are not
	available from this backend, so _forward() returns None for att_weights.
	'''

	def __init__(self, *args, **kwargs) -> None:
		'''Initialise XformerSelfAttention layers.

		Args:
			1. *args: passed to MultiHeadSelfAttentionBase.
			2. **kwargs: passed to MultiHeadSelfAttentionBase.
		'''
		super().__init__(*args, **kwargs)
		self.resid_dropout = nn.Dropout(self.dropout)
		# combine q, k, v projections for efficiency
		self.qkv_projection = nn.Linear(self.emb_dim, 3 * self.emb_dim, bias=self.bias)
		# output projection
		self.c_proj = nn.Linear(self.emb_dim, self.emb_dim, bias=self.bias)

	def _forward(self, x: Tensor, tok_mask: Tensor) -> tuple[Tensor, None]:
		'''Compute self-attention via xformers memory_efficient_attention.

		Args:
			1. x: Tensor  [float32, (B, T, D)] input embeddings.
			2. tok_mask: Tensor  [bool, (B, T)] per-token mask (not used directly by xformers; causal mask applied via LowerTriangularMask).
		Returns:
			result: tuple[Tensor, None]  (output (B, T, D), None) -- weights unavailable from xformers.
		'''
		B, T, D = x.shape
		H = self.n_heads
		# proj q, k, v for all heads
		q, k, v = self.qkv_projection(x).split(self.emb_dim, dim=2)
		# xformers expects (B, T, H, Dh) layout
		q = rearrange(q, 'B T (H Dh) -> B T H Dh', H=H)  # (B, T, H, D//H)
		k = rearrange(k, 'B T (H Dh) -> B T H Dh', H=H)  # (B, T, H, D//H)
		v = rearrange(v, 'B T (H Dh) -> B T H Dh', H=H)  # (B, T, H, D//H)
		# compute attention via xformers
		y = memory_efficient_attention(q, k, v, LowerTriangularMask(), self.dropout, None)  # (B, T, H, D//H)
		# combine heads
		y = rearrange(y, 'B T H Dh -> B T (H Dh)')  # (B, T, D)
		y = self.resid_dropout(self.c_proj(y))
		return y, None


class XformerCrossAttention(MultiHeadCrossAttentionBase):
	'''Multi-head cross-attention using xformers memory_efficient_attention backend.

	Properties:
		1. resid_dropout: nn.Dropout  dropout applied after the output projection.
		2. q_projection: nn.Linear  query linear projection (D -> D).
		3. kv_projection: nn.Linear  combined K, V linear projection (D -> 2D).
		4. c_proj: nn.Linear  output linear projection (D -> D).

	Uses xformers.ops.memory_efficient_attention which expects tensors in (B, T, H, Dh)
	layout. No causal mask is applied for cross-attention. Attention weights are not
	available from this backend, so _forward() returns None for att_weights.
	'''

	def __init__(self, *args, **kwargs) -> None:
		'''Initialise XformerCrossAttention layers.

		Args:
			1. *args: passed to MultiHeadCrossAttentionBase.
			2. **kwargs: passed to MultiHeadCrossAttentionBase.
		'''
		super().__init__(*args, **kwargs)
		self.resid_dropout = nn.Dropout(self.dropout)
		self.q_projection = nn.Linear(self.emb_dim, self.emb_dim, bias=self.bias)
		# combine k, v projections for efficiency
		self.kv_projection = nn.Linear(self.emb_dim, 2 * self.emb_dim, bias=self.bias)
		# output projection
		self.c_proj = nn.Linear(self.emb_dim, self.emb_dim, bias=self.bias)

	def _forward(self, x_q: Tensor, x_kv: Tensor, q_tok_mask: Tensor, kv_tok_mask: Tensor) -> tuple[Tensor, None]:
		'''Compute cross-attention via xformers memory_efficient_attention.

		Args:
			1. x_q: Tensor  [float32, (B, Tq, D)] query embeddings.
			2. x_kv: Tensor  [float32, (B, Tk, D)] key/value embeddings.
			3. q_tok_mask: Tensor  [bool, (B, Tq)] query token mask (not used directly by xformers).
			4. kv_tok_mask: Tensor  [bool, (B, Tk)] key/value token mask (not used directly by xformers).
		Returns:
			result: tuple[Tensor, None]  (output (B, Tq, D), None) -- weights unavailable from xformers.
		'''
		B, Tq, D = x_q.shape
		_, Tk, _ = x_kv.shape
		H = self.n_heads
		# proj query for all heads
		q = self.q_projection(x_q)
		q = rearrange(q, 'B Tq (H Dh) -> B Tq H Dh', H=H)  # (B, Tq, H, D//H)
		# proj key & value for all heads
		k, v = self.kv_projection(x_kv).split(self.emb_dim, dim=2)
		k = rearrange(k, 'B Tk (H Dh) -> B Tk H Dh', H=H)  # (B, Tk, H, D//H)
		v = rearrange(v, 'B Tk (H Dh) -> B Tk H Dh', H=H)  # (B, Tk, H, D//H)
		# compute attention via xformers (no causal mask for cross-attention)
		y = memory_efficient_attention(q, k, v, None, self.dropout, None)  # (B, Tq, H, D//H)
		# combine heads
		y = rearrange(y, 'B Tq H Dh -> B Tq (H Dh)')  # (B, Tq, D)
		y = self.resid_dropout(self.c_proj(y))
		return y, None

# Module-level aliases for lookup by transformer blocks
SELF_ATTENTION_CLS = XformerSelfAttention
CROSS_ATTENTION_CLS = XformerCrossAttention
