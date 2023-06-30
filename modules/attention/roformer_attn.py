import torch
from torch import nn
from torch import Tensor
import math
import functools

### RotaryEmbedding Helper Functions ###
def get_angles(theta, seq_len, hidden_dim):
	# angular speed
	w = 1. / (theta ** (torch.arange(1, hidden_dim+1, 2, dtype=torch.float)[:(hidden_dim // 2)] / hidden_dim))
	# time
	t = torch.arange(1, seq_len+1, dtype=torch.float)
	angles = torch.einsum('i, j -> i j', t, w)
	return angles

def get_sin_cos(angles):
	return torch.sin(angles), torch.cos(angles)

# Rotate a vector
# x1' = x1 cos - x2 sin
# x2' = x1 sin + x2 cos

def rotate_len2_subvectors(x, sin, cos):
	''' Treat the last dimension as vectors of length 2 and rotate them by the given angles.
	Inputs:
	x: (..., l, d) where d % 2 == 0
	sin: (l, d / 2)
	cos: (l, d / 2)

	Output:
	rotated: (..., l)
	  '''
	assert x.shape[-1] % 2 == 0, 'x.shape[-1] must be even'
	assert sin.shape[-1] == cos.shape[-1] == x.shape[-1] // 2, 'sin.shape[-1] must equal cos.shape[-1] and x.shape[-1] // 2'
	x1 = x[..., ::2]
	x2 = x[..., 1::2]
	x1_prime = x1 * cos - x2 * sin
	x2_prime = x1 * sin + x2 * cos
	return torch.stack((x1_prime, x2_prime), dim=-1).flatten(-2)

class RotaryEmbedding(nn.Module):

	''' Rotary positional embedding.
	Implements the rotary positional embedding from https://arxiv.org/abs/2104.09864.
	Applies rotary embeddings to the given key or query tensor in an attention module.

	Inputs:
		`x`: Tensor<Float>[B, T, C] input tensor (key or query).
	
	Outputs:
		Tensor<Float>[B, T, C] output tensor (rotated key or query).

	Notes:
		Validated against the implementation at: https://github.com/lucidrains/rotary-embedding-torch/
		`
		B = 1024
		H = 8
		T = 128
		D = 64
		k = torch.randn((B, H, T, D), dtype=torch.float, device='cuda:0')
		out1 = rotary_embedding(k)
		out2 = rotary_embedding_lucid.rotate_queries_or_keys(k)
		print('Max Diff:', (out1 - out2).abs().max().item())
		print('Mean Diff:', (out1 - out2).abs().mean().item())
		print('Std Diff:', (out1 - out2).abs().std().item())
		`
		Result:
			Max Diff: 9.5367431640625e-07
			Mean Diff: 9.750485752135774e-09
			Std Diff: 3.2006010286522724e-08
	'''

	@staticmethod
	@functools.cache
	def get_sin_cos(theta, hidden_dim, dtype):
		''' Get the sin/cos tables for the given theta and hidden_dim. '''
		if not hasattr(RotaryEmbedding, 'sin') or not hasattr(RotaryEmbedding, 'cos'):
			MAX_SEQ_LEN = 8192 # we hardcode a really big number here to avoid recomputing the sin/cos tables
			angles = get_angles(theta, MAX_SEQ_LEN, hidden_dim)
			sin, cos = get_sin_cos(angles)
		return sin.type(dtype), cos.type(dtype)


	def __init__(self, theta, hidden_dim):
		super().__init__()
		self.theta = theta
		self.hidden_dim = hidden_dim

	def forward(self, x):
		# get sequence length
		T = x.shape[-2]
		# get sin/cos tables
		sin, cos = RotaryEmbedding.get_sin_cos(self.theta, self.hidden_dim, x.dtype)
		# trim to the correct sequence length
		sin = sin[:T, :].to(x.device)
		cos = cos[:T, :].to(x.device)
		# rotate x
		return rotate_len2_subvectors(x, sin, cos)

### Attention Modules ###
class MultiHeadSelfAttention(nn.Module):

	''' Multi-head self attention.
	Implements a somewhat optimized version of the self attention by combining the q, k, v projections.
	
	Inputs:
		`x`: Tensor<Float>[B, T, C] input tensor.
		`tok_mask`: Tensor<Bool>[B, T] per-token mask applied to the `x`, false is masked out, true is preserved - masks both keys and queries.

	Outputs:
		Tensor<Float>[B, T, C] output tensor.
	'''

	def __init__(self, n_heads: int, emb_dim: int, dropout: float, bias: bool = False, is_causal: bool = False):
		super().__init__()
		self.is_causal = is_causal
		self.n_heads = n_heads
		self.emb_dim = emb_dim
		self.attn_dropout = nn.Dropout(dropout)
		self.resid_dropout = nn.Dropout(dropout)
		# combine q, k, v projections for efficiency
		self.qkv_projection = nn.Linear(emb_dim, 3 * emb_dim, bias=bias)
		# output projection
		self.c_proj = nn.Linear(emb_dim, emb_dim, bias=bias)
		self.rotary_embedding = RotaryEmbedding(theta=10000, hidden_dim=emb_dim // n_heads)

	def forward(self, x: Tensor, tok_mask: Tensor):
		B, T, C = x.shape
		# proj q, k, v for all heads
		# the heads are treated as a batch dimension
		q, k, v = self.qkv_projection(x).split(self.emb_dim, dim=2)
		q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
		k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
		v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
		# apply rotary embedding to q, k
		q = self.rotary_embedding(q)
		k = self.rotary_embedding(k)
		# compute attention
		att_weights = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
		mask = tok_mask.view(B, 1, T) # (B, 1, T)
		mask = mask.tile(1, T, 1) # (B, T, T)
		mask = mask & mask.transpose(-2, -1) # (B, T, T)
		mask = mask.view(B, 1, T, T) # (B, 1, T, T)
		if self.is_causal:
			causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
			mask = mask & causal_mask[None, None, :, :]
		att_weights = att_weights.masked_fill(mask == 0, -1e9)
		att_weights = nn.functional.softmax(att_weights, dim=-1)
		att_weights = self.attn_dropout(att_weights)
		y = att_weights @ v
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

	def __init__(self, n_heads: int, emb_dim: int, dropout: float, bias: bool = False):
		super().__init__()
		self.n_heads = n_heads
		self.emb_dim = emb_dim
		self.attn_dropout = nn.Dropout(dropout)
		self.resid_dropout = nn.Dropout(dropout)
		self.q_projection = nn.Linear(emb_dim, emb_dim, bias=bias)
		# combine k, v projections for efficiency
		self.kv_projection = nn.Linear(emb_dim, 2 * emb_dim, bias=bias)
		# output projection
		self.c_proj = nn.Linear(emb_dim, emb_dim, bias=bias)
		self.rotary_embedding = RotaryEmbedding(theta=10000, hidden_dim=emb_dim // n_heads)

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
		# apply rotary embedding to q, k
		q = self.rotary_embedding(q)
		k = self.rotary_embedding(k)
		# compute attention
		att_weights = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
		# merge masks
		q_tok_mask = q_tok_mask.unsqueeze(2) # (N, T_q, 1)
		kv_tok_mask = kv_tok_mask.unsqueeze(1) # (N, 1, T_kv)
		attn_mask = q_tok_mask & kv_tok_mask
		# apply mask
		att_weights = att_weights.masked_fill(attn_mask.unsqueeze(1) == 0, -1e9)
		att_weights = nn.functional.softmax(att_weights, dim=-1)
		att_weights = self.attn_dropout(att_weights)
		y = att_weights @ v
		# combine heads
		y = y.transpose(1, 2).contiguous().view(B, T_q, C)
		y = self.resid_dropout(self.c_proj(y))
		return y