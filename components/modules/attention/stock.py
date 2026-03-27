import torch
from torch import nn, Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel
from einops import rearrange

from components.modules.attention.base import MultiHeadSelfAttentionBase, MultiHeadCrossAttentionBase


class StockSelfAttention(MultiHeadSelfAttentionBase):
	'''Multi-head self-attention using PyTorch scaled_dot_product_attention with configurable SDPA backend.

	Attributes:
		p_dropout: ``float``: dropout probability passed to SDPA.
		qkv_projection: ``nn.Linear``: combined Q, K, V linear projection (D -> 3D).
		c_proj: ``nn.Linear``: output linear projection (D -> D).
		resid_dropout: ``nn.Dropout``: dropout applied after the output projection.

	Does not support outputting attention weights (SDPA backends do not expose them).
	'''

	def __init__(self, *args, **kwargs) -> None:
		'''Initialise StockSelfAttention layers.

		Args:
			*args: passed to MultiHeadSelfAttentionBase.
			**kwargs: passed to MultiHeadSelfAttentionBase.
		'''
		super().__init__(*args, **kwargs)
		if self.output_attention:
			raise ValueError(
				f'{self.__class__.__name__} does not support output_attention=True '
				'because SDPA backends do not expose attention weights.')
		self.p_dropout = self.dropout
		self.qkv_projection = nn.Linear(self.emb_dim, 3 * self.emb_dim, bias=self.bias)
		self.c_proj = nn.Linear(self.emb_dim, self.emb_dim, bias=self.bias)
		self.resid_dropout = nn.Dropout(self.dropout)

	def get_sdp_backends(self) -> list[SDPBackend]:
		'''Return the list of SDPA backends to enable.

		Returns:
			``list[SDPBackend]``: backends to use for scaled_dot_product_attention.
		'''
		return [SDPBackend.MATH]

	def _forward(self, x: Tensor, tok_mask: Tensor) -> tuple[Tensor, None]:
		'''Compute self-attention via PyTorch scaled_dot_product_attention.

		Args:
			x: ``Tensor[(B, T, D), float32]``: input embeddings.
			tok_mask: ``Tensor[(B, T), bool]``: per-token mask; False is masked out, True is preserved.

		Returns:
			``tuple[Tensor, None]``: (output (B, T, D), None) — weights unavailable from SDPA.
		'''
		B, T, D = x.shape
		H = self.n_heads
		q, k, v = self.qkv_projection(x).split(self.emb_dim, dim=2)
		q = rearrange(q, 'B T (H Dh) -> B H T Dh', H=H)  # (B, H, T, D//H)
		k = rearrange(k, 'B T (H Dh) -> B H T Dh', H=H)
		v = rearrange(v, 'B T (H Dh) -> B H T Dh', H=H)
		# build float mask: True positions → 0.0, False positions → -inf
		mask = tok_mask[:, None, :] & tok_mask[:, :, None]  # (B, T, T)
		if self.is_causal:
			causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
			mask = mask & causal
		attn_mask = torch.zeros_like(mask, dtype=q.dtype)
		attn_mask.masked_fill_(~mask, float('-inf'))
		attn_mask = attn_mask.unsqueeze(1)  # (B, 1, T, T) — broadcast over heads
		# SDPA
		with sdpa_kernel(self.get_sdp_backends()):
			y = nn.functional.scaled_dot_product_attention(
				q, k, v, attn_mask=attn_mask, dropout_p=self.p_dropout if self.training else 0.0)
		y = rearrange(y, 'B H T Dh -> B T (H Dh)')  # (B, T, D)
		y = self.resid_dropout(self.c_proj(y))
		return y, None


class StockCrossAttention(MultiHeadCrossAttentionBase):
	'''Multi-head cross-attention using PyTorch scaled_dot_product_attention with configurable SDPA backend.

	Attributes:
		p_dropout: ``float``: dropout probability passed to SDPA.
		q_projection: ``nn.Linear``: query linear projection (D -> D).
		kv_projection: ``nn.Linear``: combined K, V linear projection (D -> 2D).
		c_proj: ``nn.Linear``: output linear projection (D -> D).
		resid_dropout: ``nn.Dropout``: dropout applied after the output projection.

	Does not support outputting attention weights (SDPA backends do not expose them).
	'''

	def __init__(self, *args, **kwargs) -> None:
		'''Initialise StockCrossAttention layers.

		Args:
			*args: passed to MultiHeadCrossAttentionBase.
			**kwargs: passed to MultiHeadCrossAttentionBase.
		'''
		super().__init__(*args, **kwargs)
		if self.output_attention:
			raise ValueError(
				f'{self.__class__.__name__} does not support output_attention=True '
				'because SDPA backends do not expose attention weights.')
		self.p_dropout = self.dropout
		self.q_projection = nn.Linear(self.emb_dim, self.emb_dim, bias=self.bias)
		self.kv_projection = nn.Linear(self.emb_dim, 2 * self.emb_dim, bias=self.bias)
		self.c_proj = nn.Linear(self.emb_dim, self.emb_dim, bias=self.bias)
		self.resid_dropout = nn.Dropout(self.dropout)

	def get_sdp_backends(self) -> list[SDPBackend]:
		'''Return the list of SDPA backends to enable.

		Returns:
			``list[SDPBackend]``: backends to use for scaled_dot_product_attention.
		'''
		return [SDPBackend.MATH]

	def _forward(self, x_q: Tensor, x_kv: Tensor, q_tok_mask: Tensor, kv_tok_mask: Tensor) -> tuple[Tensor, None]:
		'''Compute cross-attention via PyTorch scaled_dot_product_attention.

		Args:
			x_q: ``Tensor[(B, Tq, D), float32]``: query embeddings.
			x_kv: ``Tensor[(B, Tk, D), float32]``: key/value embeddings.
			q_tok_mask: ``Tensor[(B, Tq), bool]``: query token mask; False is masked out.
			kv_tok_mask: ``Tensor[(B, Tk), bool]``: key/value token mask; False is masked out.

		Returns:
			``tuple[Tensor, None]``: (output (B, Tq, D), None) — weights unavailable from SDPA.
		'''
		B, Tq, D = x_q.shape
		_, Tk, _ = x_kv.shape
		H = self.n_heads
		q = rearrange(self.q_projection(x_q), 'B Tq (H Dh) -> B H Tq Dh', H=H)
		k, v = self.kv_projection(x_kv).split(self.emb_dim, dim=2)
		k = rearrange(k, 'B Tk (H Dh) -> B H Tk Dh', H=H)
		v = rearrange(v, 'B Tk (H Dh) -> B H Tk Dh', H=H)
		# build float mask from boolean token masks
		mask = q_tok_mask[:, :, None] & kv_tok_mask[:, None, :]  # (B, Tq, Tk)
		attn_mask = torch.zeros_like(mask, dtype=q.dtype)
		attn_mask.masked_fill_(~mask, float('-inf'))
		attn_mask = attn_mask.unsqueeze(1)  # (B, 1, Tq, Tk)
		# SDPA
		with sdpa_kernel(self.get_sdp_backends()):
			y = nn.functional.scaled_dot_product_attention(
				q, k, v, attn_mask=attn_mask, dropout_p=self.p_dropout if self.training else 0.0)
		y = rearrange(y, 'B H Tq Dh -> B Tq (H Dh)')  # (B, Tq, D)
		y = self.resid_dropout(self.c_proj(y))
		return y, None

# Module-level aliases for lookup by transformer blocks
SELF_ATTENTION_CLS = StockSelfAttention
CROSS_ATTENTION_CLS = StockCrossAttention
