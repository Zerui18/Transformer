import torch
from torch import nn
from torch import Tensor

class MultiHeadSelfAttention(nn.Module):

	''' Multi-head self attention.
	Implements a somewhat optimized version of the self attention by combining the q, k, v projections.
	
	Inputs:
		`x`: Tensor<Float>[B, T, C] input tensor.
		`tok_mask`: Tensor<Bool>[B, T] per-token mask applied to the `x`, false is masked out, true is preserved - masks both keys and queries.

	Outputs:
		Tensor<Float>[B, T, C] output tensor.
	'''

	def get_attention_args(self):
		return {
			'enable_math': True,
			'enable_flash': False,
			'enable_mem_efficient': False
		}

	def __init__(self, n_heads: int, emb_dim: int, dropout: float, bias: bool = False, is_causal: bool = False):
		super().__init__()
		self.is_causal = is_causal
		self.p_dropout = dropout
		self.n_heads = n_heads
		# combine q, k, v projections for efficiency
		self.qkv_projection = nn.Linear(emb_dim, 3 * emb_dim, bias=bias)
		# output projection
		self.c_proj = nn.Linear(emb_dim, emb_dim, bias=bias)
		self.resid_dropout = nn.Dropout(dropout)

	def forward(self, x: Tensor, tok_mask: Tensor):
		B, T, C = x.shape
		# proj q, k, v for all heads
		# the heads are treated as a batch dimension
		q, k, v = self.qkv_projection(x).split(self.emb_dim, dim=2)
		q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
		k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
		v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
		# apply attention
		with torch.backends.cuda.sdp_kernel(**self.get_attention_args()):
			mask = tok_mask
			if self.is_causal:
				causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
				mask = mask & causal_mask[None, :, :]
			y = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.p_dropout)
		# combine heads
		y = y.transpose(1, 2).contiguous().view(B, T, C)
		y = self.resid_dropout(self.c_proj(y))
		return y

class MultiHeadCrossAttention(nn.Module):

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

	def get_attention_args(self):
		return {
			'enable_math': True,
			'enable_flash': False,
			'enable_mem_efficient': False
		}

	def __init__(self, n_heads: int, emb_dim: int, dropout: float, bias: bool = False):
		super().__init__()
		self.p_dropout = dropout
		self.n_heads = n_heads
		self.emb_dim = emb_dim
		self.q_projection = nn.Linear(emb_dim, emb_dim, bias=bias)
		# combine k, v projections for efficiency
		self.kv_projection = nn.Linear(emb_dim, 2 * emb_dim, bias=bias)
		# output projection
		self.c_proj = nn.Linear(emb_dim, emb_dim, bias=bias)
		self.resid_dropout = nn.Dropout(dropout)


	def forward(self, x_q: Tensor, x_kv: Tensor, q_tok_mask: Tensor, kv_tok_mask: Tensor):
		# proj query for all heads
		B, T_q, C = x_q.shape
		q = self.q_projection(x_q)
		q = q.view(B, T_q, self.n_heads, C // self.n_heads).transpose(1, 2)
		# proj key & value for all heads
		B, T_kv, C = x_kv.shape
		k, v = self.kv_projection(x_kv).split(self.emb_dim, dim=2)
		k = k.view(B, T_kv, self.n_heads, C // self.n_heads).transpose(1, 2)
		v = v.view(B, T_kv, self.n_heads, C // self.n_heads).transpose(1, 2)
		# apply attention
		with torch.backends.cuda.sdp_kernel(**self.get_attention_args()):
			# merge masks
			q_tok_mask = q_tok_mask.unsqueeze(2) # (N, T_q, 1)
			kv_tok_mask = kv_tok_mask.unsqueeze(1) # (N, 1, T_kv)
			attn_mask = q_tok_mask & kv_tok_mask
			is_special_attention = self.get_attention_args()['enable_math'] == False
			if is_special_attention:
				y = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.p_dropout)
			else:
				y = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask , dropout_p=self.p_dropout)
		# combine heads
		y = y.transpose(1, 2).contiguous().view(B, T_q, C)
		y = self.resid_dropout(self.c_proj(y))
		return y