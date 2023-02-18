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
		att_weights = att_weights.masked_fill(mask == 0, float('-inf'))
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
		att_weights = att_weights.masked_fill(mask == 0, float('-inf'))
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

class TransformerModelAR(nn.Module):

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
		''' Autoregressively generate the target sequence.
		
		The forward pass will step through range(1, block_size), predicting the next token at each step.
		Teacher forcing with a random probability is used (ie. the next token is either the ground truth or the predicted token).
		The entire generated sequence, in the form of logits, is returned.

		TBD:
		This approach doesn't really make sense:
		At each step, the model is required (by cross-entropy) to output the exact same token as the ground truth.
		However, the model's input at each step may not be the same as the ground truth up to that step due to teacher forcing with probability.
		Hence, it's unrealistic to expect the model to output the exact same token as the ground truth and unreasonable to penalize the model for not doing so.
		>> Instead, either set teacher forcing to always take effect, or use a different loss function that takes into account the model's input at each step.
		>> The latter approach is probably better, but it's not clear how to implement it.
		>> The next commit will implement the former approach + batching by similar sentence length.
		>> This will allow for more efficient training (less wasted computation on padding tokens of unequal length sentences).
		'''
		bs = src.size(0)
		out_encoder = self.encoder(src, src_mask)
		# initialize the decoder output with zeros
		history_logits = torch.zeros((bs, self.config.block_size, self.config.vocab_size), dtype=torch.float32, device=src.device)
		# initialize the decoder input with the <BOS> token
		x_decoder = torch.zeros((bs, self.config.block_size), dtype=torch.long, device=src.device)
		x_decoder[:, 0] = tgt[:, 0]
		# autoregressively generate the decoder output
		# the first token (<BOS>) is already known
		for t in range(1, self.config.block_size):
			out_decoder = self.decoder(out_encoder, x_decoder.clone(), src_mask, tgt_mask)
			logits = self.lm_head(out_decoder)
			# update the decoder output history
			history_logits[:, t] = logits[:, t]
			# update x_decoder with either the ground truth or the predicted token
			# with probability self.config.teacher_forcing_ratio
			# avoid data-dependent control flow to prevent graph breaks
			predicted_token = logits[:, -1].argmax(dim=-1)
			teacher_forcing_mask = (torch.rand(bs, device=src.device) < self.config.teacher_forcing_ratio).long()
			x_decoder[:, t] += teacher_forcing_mask * tgt[:, t] + (1 - teacher_forcing_mask) * predicted_token
		return history_logits

