import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from typing import *
from config import TransformerConfig

GPU = torch.device('cuda')

def new_gelu(x):
	"""
	Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
	Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
	"""
	return 0.5 * x * (1.0 + torch.tanh(torch.sqrt(2.0 / torch.pi) * (x + 0.044715 * torch.power(x, 3.0))))

class InputEmbeddings(nn.Module):
	''' Apply learnable token and position embeddings to input tokens. '''

	def __init__(self, config: TransformerConfig):
		super().__init__()
		self.token_embedding_table = nn.Embedding(config.vocab_size, config.emb_dim)
		self.position_embedding_table = nn.Embedding(config.block_size, config.emb_dim)
		self.register_buffer('pos_emb_index', torch.arange(config.block_size))
	
	def forward(self, x):
		B, T = x.shape
		tok_embd = self.token_embedding_table(x)
		pos_embd = self.position_embedding_table(self.pos_emb_index[:T])
		return tok_embd + pos_embd

class MultiHeadSelfAttention(nn.Module):

	''' Multi-head self attention.
	Implements a somewhat optimized version of the self attention by combining the q, k, v projections.
	
	Inputs:
		x: input tensor of shape (B, T, C)
		mask: mask tensor of shape broadcastable to (B, T, T)
	'''

	def __init__(self, config: TransformerConfig, is_causal: bool = False):
		super().__init__()
		self.is_causal = is_causal
		self.n_heads = config.n_heads
		self.emb_dim = config.emb_dim
		# combine q, k, v projections for efficiency
		self.qkv_projection = nn.Linear(config.emb_dim, 3 * config.emb_dim, bias=config.bias)
		self.register_buffer('scale', torch.tensor(config.emb_dim ** -0.5, dtype=torch.float32))
		if is_causal:
			self.register_buffer('causal_mask', torch.tril(torch.ones(1, config.block_size, config.block_size, dtype=torch.bool)))

	def forward(self, x, mask):
		B, T, C = x.shape
		# proj q, k, v for all heads
		# the heads are treated as a batch dimension
		q, k, v = self.qkv_projection(x).split(self.emb_dim, dim=2)
		q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
		k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
		v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
		# compute attention
		att_weights = (q @ k.transpose(-2, -1)) / self.scale
		mask = mask.unsqueeze(1) # apply mask over all heads
		if self.is_causal:
			mask = mask & self.causal_mask[:, :T, :T]
		att_weights = att_weights.masked_fill(mask == 0, -1e9)
		att_weights = F.softmax(att_weights, dim=-1)
		y = att_weights @ v
		# combine heads
		y = y.transpose(1, 2).contiguous().view(B, T, C)
		return y

class MultiHeadCrossAttention(nn.Module):

	''' Multi-head cross attention.
	Implements a somewhat optimized version of the cross attention by combining the k, v projections.
	
	Inputs:
		x_q: query tensor of shape (B, T, C)
		x_kv: key and value tensor of shape (B, T, C)
		mask: mask tensor of shape broadcastable to (B, T, T)
	'''

	def __init__(self, config: TransformerConfig):
		super().__init__()
		self.n_heads = config.n_heads
		self.emb_dim = config.emb_dim
		self.q_projection = nn.Linear(config.emb_dim, config.emb_dim, bias=config.bias)
		# combine k, v projections for efficiency
		self.kv_projection = nn.Linear(config.emb_dim, 2 * config.emb_dim, bias=config.bias)
		self.register_buffer('scale', torch.tensor(config.emb_dim ** -0.5, dtype=torch.float32))

	def forward(self, x_q, x_kv, mask):
		# proj query for all heads
		B, T, C = x_q.shape
		q = self.q_projection(x_q)
		q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
		# proj key & value for all heads
		B, T, C = x_kv.shape
		k, v = self.kv_projection(x_kv).split(self.emb_dim, dim=2)
		k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
		v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
		# compute attention
		att_weights = (q @ k.transpose(-2, -1)) / self.scale
		mask = mask.unsqueeze(1) # apply mask over all heads
		att_weights = att_weights.masked_fill(mask == 0, -1e9)
		att_weights = F.softmax(att_weights, dim=-1)
		y = att_weights @ v
		# combine heads
		y = y.transpose(1, 2).contiguous().view(B, T, C)
		return y

class FeedFoward(nn.Module):

	def __init__(self, config: TransformerConfig):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(config.emb_dim, 4 * config.emb_dim),
			nn.ReLU(),
			nn.Linear(4 * config.emb_dim, config.emb_dim),
		)
		if config.dropout:
			self.net.append(nn.Dropout(config.dropout))

	def forward(self, x):
		return self.net(x)

class EncoderBlock(nn.Module):

	def __init__(self, config: TransformerConfig):
		super().__init__()
		self.sa_module = MultiHeadSelfAttention(config)
		self.fw_module = FeedFoward(config)
		self.ln1 = nn.LayerNorm(config.emb_dim)
		self.ln2 = nn.LayerNorm(config.emb_dim)
	
	def forward(self, src, src_mask):
		x = src + self.sa_module(self.ln1(src), src_mask)
		x = x + self.fw_module(self.ln2(src))
		return x

class Encoder(nn.Module):

	def __init__(self, config: TransformerConfig):
		super().__init__()
		self.input_embeddings = InputEmbeddings(config)
		self.blocks = nn.ModuleList([EncoderBlock(config) for _ in range(config.n_blocks)])
		self.use_grad_ckpt = config.use_grad_ckpt
	
	def forward(self, src, src_mask):
		x = self.input_embeddings(src)
		for block in self.blocks:
			if self.use_grad_ckpt:
				forward = lambda *inputs: block(*inputs)
				x = checkpoint(forward, x, src_mask, preserve_rng_state=False)
			else:
				x = block(x, src_mask)
		return x

class DecoderBlock(nn.Module):

	def __init__(self, config: TransformerConfig):
		super().__init__()
		self.sa_module = MultiHeadSelfAttention(config, is_causal=True)
		self.ca_module = MultiHeadCrossAttention(config)
		self.fw_module = FeedFoward(config)
		self.ln1 = nn.LayerNorm(config.emb_dim)
		self.ln2 = nn.LayerNorm(config.emb_dim)
		self.ln3 = nn.LayerNorm(config.emb_dim)
	
	def forward(self, src, tgt, src_mask, tgt_mask):
		x = tgt + self.sa_module(self.ln1(tgt), tgt_mask)
		x = x + self.ca_module(self.ln2(src), self.ln2(tgt), src_mask)
		x = x + self.fw_module(self.ln3(x))
		return x

class Decoder(nn.Module):

	def __init__(self, config: TransformerConfig):
		super().__init__()
		self.input_embeddings = InputEmbeddings(config)
		self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_blocks)])
		self.use_grad_ckpt = config.use_grad_ckpt
	
	def forward(self, src, tgt, src_mask, tgt_mask):
		x = self.input_embeddings(tgt)
		for block in self.blocks:
			if self.use_grad_ckpt:
				forward = lambda *inputs: block(*inputs)
				x = checkpoint(forward, src, x, src_mask, tgt_mask, preserve_rng_state=False)
			else:
				x = block(src, x, src_mask, tgt_mask)
		return x

class LMHead(nn.Module):

	def __init__(self, config: TransformerConfig):
		super().__init__()
		self.ln = nn.LayerNorm(config.emb_dim)
		self.logits_head = nn.Linear(config.emb_dim, config.vocab_size)
	
	def forward(self, x):
		x = self.ln(x)
		x = self.logits_head(x)
		return x

class TransformerModel(nn.Module):

	def __init__(self, config: TransformerConfig):
		super().__init__()
		self.config = config
		# model parts
		self.encoder = Encoder(config)
		self.decoder = Decoder(config)
		self.lm_head = LMHead(config)
		# weight tying
		if config.weight_tying:
			self.decoder.input_embeddings.token_embedding_table.weight = self.lm_head.logits_head.weight
	
	def forward(self, src, tgt, src_mask, tgt_mask):
		enc = self.encoder(src, src_mask)
		dec = self.decoder(enc, tgt, src_mask, tgt_mask)
		logits = self.lm_head(dec)
		return logits

