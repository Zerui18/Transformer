import torch
from torch import nn
from torch import Tensor
from xformers.ops import memory_efficient_attention, LowerTriangularMask
from .base import MultiHeadSelfAttentionBase, MultiHeadCrossAttentionBase

class MultiHeadSelfAttention(MultiHeadSelfAttentionBase):

	''' Multi-head self attention.
	Implements a somewhat optimized version of the self attention by combining the q, k, v projections.
	
	Inputs:
		`x`: Tensor<Float>[B, T, C] input tensor.
		`tok_mask`: Tensor<Bool>[B, T] per-token mask applied to the `x`, false is masked out, true is preserved - masks both keys and queries.

	Outputs:
		Tensor<Float>[B, T, C] output tensor.
	'''

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.resid_dropout = nn.Dropout(self.dropout)
		# combine q, k, v projections for efficiency
		self.qkv_projection = nn.Linear(self.emb_dim, 3 * self.emb_dim, bias=self.bias)
		# output projection
		self.c_proj = nn.Linear(self.emb_dim, self.emb_dim, bias=self.bias)

	def _forward(self, x: Tensor, tok_mask: Tensor):
		B, T, C = x.shape
		# proj q, k, v for all heads
		# the heads are treated as a batch dimension
		q, k, v = self.qkv_projection(x).split(self.emb_dim, dim=2)
		q = q.view(B, T, self.n_heads, C // self.n_heads)
		k = k.view(B, T, self.n_heads, C // self.n_heads)
		v = v.view(B, T, self.n_heads, C // self.n_heads)
		# compute attention
		y = memory_efficient_attention(q, k, v, LowerTriangularMask(), self.dropout, None)
		# combine heads
		y = y.contiguous().view(B, T, C)
		y = self.resid_dropout(self.c_proj(y))
		return y, None

class MultiHeadCrossAttention(MultiHeadCrossAttentionBase):

	''' Multi-head cross attention.
	Implements a somewhat optimized version of the cross attention by combining the k, v projections.
	
	Inputs:
		`x_q`: Tensor<Float>[B, T_q, C] query input tensor.
		`x_kv`: Tensor<Float>[B, T_kv, C] key and value input tensor.
		`q_tok_mask`: Tensor<Bool>[B, T_q] mask applied to the `x_q`, false is masked out, true is preserved - applies to q only.
		`kv_tok_mask`: Tensor<Bool>[B, T_kv] mask applied to the `x_kv`, false is masked out, true is preserved - applies to k and v.

	Outputs:
		Tensor<Float>[B, T_q, C] output tensor.
	'''

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.resid_dropout = nn.Dropout(self.dropout)
		self.q_projection = nn.Linear(self.emb_dim, self.emb_dim, bias=self.bias)
		# combine k, v projections for efficiency
		self.kv_projection = nn.Linear(self.emb_dim, 2 * self.emb_dim, bias=self.bias)
		# output projection
		self.c_proj = nn.Linear(self.emb_dim, self.emb_dim, bias=self.bias)

	def _forward(self, x_q: Tensor, x_kv: Tensor, q_tok_mask: Tensor, kv_tok_mask: Tensor):
		# proj query for all heads
		B, T_q, C = x_q.shape
		q = self.q_projection(x_q)
		q = q.view(B, T_q, self.n_heads, C // self.n_heads)
		# proj key & value for all heads
		B, T_kv, C = x_kv.shape
		k, v = self.kv_projection(x_kv).split(self.emb_dim, dim=2)
		k = k.view(B, T_kv, self.n_heads, C // self.n_heads)
		v = v.view(B, T_kv, self.n_heads, C // self.n_heads)
		# compute attention
		y = memory_efficient_attention(q, k, v, None, self.dropout, None)
		# combine heads
		y = y.contiguous().view(B, T_q, C)
		y = self.resid_dropout(self.c_proj(y))
		return y, None