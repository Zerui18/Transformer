import torch
from torch import nn, Tensor
from einops import rearrange

from components.modules.attention.base import MultiHeadSelfAttentionBase, MultiHeadCrossAttentionBase


class StockSelfAttention(MultiHeadSelfAttentionBase):
	'''Multi-head self-attention using PyTorch scaled_dot_product_attention with configurable SDPA backend.

	Properties:
		1. p_dropout: float  dropout probability passed to SDPA.
		2. qkv_projection: nn.Linear  combined Q, K, V linear projection (D -> 3D).
		3. c_proj: nn.Linear  output linear projection (D -> D).
		4. resid_dropout: nn.Dropout  dropout applied after the output projection.

	Uses torch.nn.functional.scaled_dot_product_attention with a selectable backend
	via get_attention_args(). Does not support outputting attention weights (SDPA
	backends do not expose them); output_attention must remain False.
	'''

	def __init__(self, *args, **kwargs) -> None:
		'''Initialise StockSelfAttention layers.

		Args:
			1. *args: passed to MultiHeadSelfAttentionBase.
			2. **kwargs: passed to MultiHeadSelfAttentionBase.

		output_attention is forced to False because SDPA backends do not return attention weights.
		'''
		super().__init__(*args, **kwargs)
		if self.output_attention:
			raise ValueError(
				f'{self.__class__.__name__} does not support output_attention=True '
				'because SDPA backends do not expose attention weights.'
			)
		self.p_dropout = self.dropout
		# combine q, k, v projections for efficiency
		self.qkv_projection = nn.Linear(self.emb_dim, 3 * self.emb_dim, bias=self.bias)
		# output projection
		self.c_proj = nn.Linear(self.emb_dim, self.emb_dim, bias=self.bias)
		self.resid_dropout = nn.Dropout(self.dropout)

	def get_attention_args(self) -> dict[str, bool]:
		'''Return SDPA backend selection flags for torch.backends.cuda.sdp_kernel.

		Returns:
			args: dict[str, bool]  keys are enable_math, enable_flash, enable_mem_efficient.
		'''
		return {
			'enable_math': True,
			'enable_flash': False,
			'enable_mem_efficient': False,
		}

	def _forward(self, x: Tensor, tok_mask: Tensor) -> tuple[Tensor, None]:
		'''Compute self-attention via PyTorch scaled_dot_product_attention.

		Args:
			1. x: Tensor  [float32, (B, T, D)] input embeddings.
			2. tok_mask: Tensor  [bool, (B, T)] per-token mask; False is masked out, True is preserved.
		Returns:
			result: tuple[Tensor, None]  (output (B, T, D), None) -- weights unavailable from SDPA.
		'''
		B, T, D = x.shape
		H = self.n_heads
		# proj q, k, v for all heads
		q, k, v = self.qkv_projection(x).split(self.emb_dim, dim=2)
		q = rearrange(q, 'B T (H Dh) -> B H T Dh', H=H)  # (B, H, T, D//H)
		k = rearrange(k, 'B T (H Dh) -> B H T Dh', H=H)  # (B, H, T, D//H)
		v = rearrange(v, 'B T (H Dh) -> B H T Dh', H=H)  # (B, H, T, D//H)
		# construct attention mask
		with torch.backends.cuda.sdp_kernel(**self.get_attention_args()):
			if self.is_causal:
				causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
				mask = tok_mask.unsqueeze(-1) & causal_mask.unsqueeze(0)  # (B, T, T)
			else:
				mask = torch.einsum('bt,bT->btT', tok_mask.float(), tok_mask.float()) > 0  # (B, T, T)
			mask = mask.unsqueeze(1).tile(1, H, 1, 1)  # (B, H, T, T)
			y = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=self.is_causal)
		# combine heads
		y = rearrange(y, 'B H T Dh -> B T (H Dh)')  # (B, T, D)
		y = self.resid_dropout(self.c_proj(y))
		return y, None


class StockCrossAttention(MultiHeadCrossAttentionBase):
	'''Multi-head cross-attention using PyTorch scaled_dot_product_attention with configurable SDPA backend.

	Properties:
		1. p_dropout: float  dropout probability passed to SDPA.
		2. q_projection: nn.Linear  query linear projection (D -> D).
		3. kv_projection: nn.Linear  combined K, V linear projection (D -> 2D).
		4. c_proj: nn.Linear  output linear projection (D -> D).
		5. resid_dropout: nn.Dropout  dropout applied after the output projection.

	Uses torch.nn.functional.scaled_dot_product_attention with a selectable backend
	via get_attention_args(). Does not support outputting attention weights (SDPA
	backends do not expose them); output_attention must remain False.
	'''

	def __init__(self, *args, **kwargs) -> None:
		'''Initialise StockCrossAttention layers.

		Args:
			1. *args: passed to MultiHeadCrossAttentionBase.
			2. **kwargs: passed to MultiHeadCrossAttentionBase.

		output_attention is forced to False because SDPA backends do not return attention weights.
		'''
		super().__init__(*args, **kwargs)
		if self.output_attention:
			raise ValueError(
				f'{self.__class__.__name__} does not support output_attention=True '
				'because SDPA backends do not expose attention weights.'
			)
		self.p_dropout = self.dropout
		self.q_projection = nn.Linear(self.emb_dim, self.emb_dim, bias=self.bias)
		# combine k, v projections for efficiency
		self.kv_projection = nn.Linear(self.emb_dim, 2 * self.emb_dim, bias=self.bias)
		# output projection
		self.c_proj = nn.Linear(self.emb_dim, self.emb_dim, bias=self.bias)
		self.resid_dropout = nn.Dropout(self.dropout)

	def get_attention_args(self) -> dict[str, bool]:
		'''Return SDPA backend selection flags for torch.backends.cuda.sdp_kernel.

		Returns:
			args: dict[str, bool]  keys are enable_math, enable_flash, enable_mem_efficient.
		'''
		return {
			'enable_math': True,
			'enable_flash': False,
			'enable_mem_efficient': False,
		}

	def _forward(self, x_q: Tensor, x_kv: Tensor, q_tok_mask: Tensor, kv_tok_mask: Tensor) -> tuple[Tensor, None]:
		'''Compute cross-attention via PyTorch scaled_dot_product_attention.

		Args:
			1. x_q: Tensor  [float32, (B, Tq, D)] query embeddings.
			2. x_kv: Tensor  [float32, (B, Tk, D)] key/value embeddings.
			3. q_tok_mask: Tensor  [bool, (B, Tq)] query token mask; False is masked out.
			4. kv_tok_mask: Tensor  [bool, (B, Tk)] key/value token mask; False is masked out.
		Returns:
			result: tuple[Tensor, None]  (output (B, Tq, D), None) -- weights unavailable from SDPA.
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
		# apply attention via SDPA
		with torch.backends.cuda.sdp_kernel(**self.get_attention_args()):
			# merge masks
			q_mask = q_tok_mask.unsqueeze(2)   # (B, Tq, 1)
			kv_mask = kv_tok_mask.unsqueeze(1)  # (B, 1, Tk)
			attn_mask = q_mask & kv_mask        # (B, Tq, Tk)
			is_special_attention = self.get_attention_args()['enable_math'] == False
			if is_special_attention:
				y = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None)
			else:
				attn_mask = attn_mask.unsqueeze(1).tile(1, H, 1, 1)  # (B, H, Tq, Tk)
				y = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.p_dropout)
		# combine heads
		y = rearrange(y, 'B H Tq Dh -> B Tq (H Dh)')  # (B, Tq, D)
		y = self.resid_dropout(self.c_proj(y))
		return y, None

# Module-level aliases for lookup by transformer blocks
SELF_ATTENTION_CLS = StockSelfAttention
CROSS_ATTENTION_CLS = StockCrossAttention
