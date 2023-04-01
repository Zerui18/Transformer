import torch
from torch import nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from .attention import MultiHeadSelfAttention, MultiHeadCrossAttention

class TransformerFeedFoward(nn.Module):

	def __init__(self, emb_dim: int, dropout: float):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(emb_dim, 4 * emb_dim),
			nn.GELU(approximate='tanh'),
			nn.Linear(4 * emb_dim, emb_dim),
		)
		if dropout:
			self.net.append(nn.Dropout(dropout))

	def forward(self, x: Tensor):
		return self.net(x)

class TransformerEncoderBlock(nn.Module):

	def __init__(self, n_heads: int, emb_dim: int, dropout: float, bias: bool = False):
		super().__init__()
		self.sa_module = MultiHeadSelfAttention(n_heads, emb_dim, dropout, bias)
		self.fw_module = TransformerFeedFoward(emb_dim, dropout)
		self.ln1 = nn.LayerNorm(emb_dim)
		self.ln2 = nn.LayerNorm(emb_dim)

	
	def forward(self, src: Tensor, src_mask: Tensor):
		x = src + self.sa_module(self.ln1(src), src_mask)
		x = x + self.fw_module(self.ln2(src))
		return x

class TransformerEncoder(nn.Module):

	def __init__(self, n_blocks: int, n_heads: int, emb_dim: int, dropout: float, bias: bool = False, use_grad_ckpt: bool = False):
		super().__init__()
		self.blocks = nn.ModuleList([TransformerEncoderBlock(n_heads, emb_dim, dropout, bias) for _ in range(n_blocks)])
		self.use_grad_ckpt = use_grad_ckpt
	
	def forward(self, src: Tensor, src_mask: Tensor):
		x = src
		for block in self.blocks:
			if self.use_grad_ckpt:
				forward = lambda *inputs: block(*inputs)
				x = checkpoint(forward, x, src_mask, preserve_rng_state=False)
			else:
				x = block(x, src_mask)
		return x

class TransformerDecoderBlock(nn.Module):

	def __init__(self, n_heads: int, emb_dim: int, dropout: float, bias: bool = False):
		super().__init__()
		self.sa_module = MultiHeadSelfAttention(n_heads, emb_dim, dropout, bias, is_causal=True)
		self.ca_module = MultiHeadCrossAttention(n_heads, emb_dim, dropout, bias)
		self.fw_module = TransformerFeedFoward(emb_dim, dropout)
		self.ln1 = nn.LayerNorm(emb_dim)
		self.ln2 = nn.LayerNorm(emb_dim)
		self.ln3 = nn.LayerNorm(emb_dim)
	
	def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor, tgt_mask: Tensor):
		x = tgt + self.sa_module(self.ln1(tgt), tgt_mask)
		x = x + self.ca_module(self.ln2(tgt), self.ln2(src), tgt_mask, src_mask)
		x = x + self.fw_module(self.ln3(x))
		return x

class TransformerDecoder(nn.Module):

	def __init__(self, n_blocks: int, n_heads: int, emb_dim: int, dropout: float, bias: bool = False, use_grad_ckpt: bool = False):
		super().__init__()
		self.blocks = nn.ModuleList([TransformerDecoderBlock(n_heads, emb_dim, dropout, bias) for _ in range(n_blocks)])
		self.use_grad_ckpt = use_grad_ckpt
	
	def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor, tgt_mask: Tensor):
		x = tgt
		for block in self.blocks:
			if self.use_grad_ckpt:
				forward = lambda *inputs: block(*inputs)
				x = checkpoint(forward, src, x, src_mask, tgt_mask, preserve_rng_state=False)
			else:
				x = block(src, x, src_mask, tgt_mask)
		return x

class TransformerLMHead(nn.Module):

	def __init__(self, emb_dim: int, tgt_vocab_size: int):
		super().__init__()
		self.ln = nn.LayerNorm(emb_dim)
		self.logits_head = nn.Linear(emb_dim, tgt_vocab_size)
	
	def forward(self, x: Tensor):
		x = self.ln(x)
		x = self.logits_head(x)
		return x