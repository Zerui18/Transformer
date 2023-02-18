import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import *
from config import TransformerConfig

def new_gelu(x):
	"""
	Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
	Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
	"""
	return 0.5 * x * (1.0 + torch.tanh(torch.sqrt(2.0 / torch.pi) * (x + 0.044715 * torch.power(x, 3.0))))

class MultiHeadSelfAttention(nn.Module):

	def __init__(self, config: TransformerConfig, is_causal: bool):
		super().__init__()
		self.config = config
		self.is_causal = is_causal
		# combine q, k, v projections for efficiency
		self.qkv_projection = nn.Linear(config.emb_dim, 3 * config.emb_dim, bias=config.bias)
		if is_causal:
			self.register_buffer('mask', torch.tril(torch.ones(config.block_size, config.block_size)))

	def forward(self, x):
		config = self.config
		B, T, C = x.shape
		# proj q, k, v
		q, k, v = self.qkv_projection(x).split(config.emb_dim, dim=2) # (B, T, config.emb_dim)
		q = q.view(B, T, config.n_heads, C // config.n_heads).transpose(1, 2) # (B, config.n_heads, T, config.emb_dim)
		k = k.view(B, T, config.n_heads, C // config.n_heads).transpose(1, 2) # (B, config.n_heads, T, config.emb_dim)
		v = v.view(B, T, config.n_heads, C // config.n_heads).transpose(1, 2) # (B, config.n_heads, T, config.emb_dim)
		# apply
		att_weights = (q @ k.transpose(-2, -1)) / torch.sqrt(config.emb_dim)
		if self.is_causal:
			att_weights = att_weights.masked_fill(self.mask[:T,:T] == 0, float('-inf'))
		att_weights = F.softmax(att_weights, dim=-1)
		y = att_weights @ v
		return y

class MultiHeadCrossAttention(nn.Module):

	def __init__(self, config: TransformerConfig):
		super().__init__()
		self.config = config
		self.q_projection = nn.Linear(config.emb_dim, config.emb_dim, bias=config.bias)
		# combine k, v projections for efficiency
		self.kv_projection = nn.Linear(config.emb_dim, 2 * config.emb_dim, bias=config.bias)

	def forward(self, x_q, x_kv):
		config = self.config
		# query
		B, T, C = x_q.shape
		q = self.q_projection(x_q) # (B, T, config.emb_dim)
		q = q.view(B, T, config.n_heads, C // config.n_heads).transpose(1, 2) # (B, config.n_heads, T, config.emb_dim)
		# key & value
		B, T, C = x_kv.shape
		k, v = self.kv_projection(x_kv).split(config.emb_dim, dim=2) # (B, T, config.emb_dim)
		k = k.view(B, T, config.n_heads, C // config.n_heads).transpose(1, 2) # (B, config.n_heads, T, config.emb_dim)
		v = v.view(B, T, config.n_heads, C // config.n_heads).transpose(1, 2) # (B, config.n_heads, T, config.emb_dim)
		# apply
		att_weights = (q @ k.transpose(-2, -1)) / torch.sqrt(config.emb_dim)
		att_weights = F.softmax(att_weights, dim=-1)
		y = att_weights @ v
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

class EncoderBlock(nn.Module):

	def __init__(self, config: TransformerConfig):
		super().__init__()
		self.sa_module = MultiHeadSelfAttention(config, is_causal=False)
		self.fw_module = FeedFoward(config)
		self.ln1 = nn.LayerNorm(config.emb_dim)
		self.ln2 = nn.LayerNorm(config.emb_dim)
	
	def forward(self, x):
		print('hello', x.device)
		# x + ... implements residual connections
		x = x + self.sa_module(self.ln1(x))
		x = x + self.fw_module(self.ln2(x))
		return x

class Encoder(nn.Module):

	def __init__(self, config: TransformerConfig):
		super().__init__()
		self.input_embeddings = InputEmbeddings(config)
		self.blocks = [EncoderBlock(config) for _ in range(config.n_blocks)]
		self.use_grad_ckpt = config.use_grad_ckpt
	
	def forward(self, x):
		x = self.input_embeddings(x)
		for block in self.blocks:
			if self.use_grad_ckpt:
				forward = lambda *inputs: block(*inputs)
				x = torch.utils.checkpoint.checkpoint(forward, x, preserve_rng_state=False)
			else:
				x = block(x)
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
	
	def forward(self, x_q, x_kv):
		# x + ... implements residual connections
		x_kv = x_kv + self.sa_module(self.ln1(x_kv))
		y = x_kv + self.ca_module(self.ln2(x_q), self.ln2(x_kv))
		y = y + self.fw_module(self.ln3(y))
		return y

class Decoder(nn.Module):

	def __init__(self, config: TransformerConfig):
		super().__init__()
		self.input_embeddings = InputEmbeddings(config)
		self.blocks = [DecoderBlock(config) for _ in range(config.n_blocks)]
		self.use_grad_ckpt = config.use_grad_ckpt
	
	def forward(self, out_encoder, x):
		x = self.input_embeddings(x)
		for block in self.blocks:
			if self.use_grad_ckpt:
				forward = lambda *inputs: block(*inputs)
				x = torch.utils.checkpoint.checkpoint(forward, out_encoder, x, preserve_rng_state=False)
			else:
				x = block(out_encoder, x)
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
	
	def forward(self, x_src, x_dst):
		bs = x_src.size(0)
		out_encoder = self.encoder(x_src)
		# initialize the decoder output with zeros
		history_out_decoder = torch.zeros((bs, self.config.block_size, self.config.emb_dim), dtype=torch.float32, device=x_src.device)
		# initialize the decoder input with the <BOS> token
		x_decoder = torch.zeros((bs, self.config.block_size), dtype=torch.long, device=x_src.device)
		x_decoder[:, 0] = x_dst[:, :1]
		# autoregressively generate the decoder output
		# the first token (<BOS>) is already known
		for t in range(1, self.config.block_size):
			out_decoder = self.decoder(out_encoder, x_decoder)
			logits = self.lm_head(out_decoder)
			# update the decoder output history
			history_out_decoder[:, t] = out_decoder[:, t]
			# update x_decoder with either the ground truth or the predicted token
			# with probability self.config.teacher_forcing_ratio
			# avoid data-dependent control flow to prevent graph breaks
			predicted_token = logits.argmax(dim=-1)
			teacher_forcing_mask = (torch.rand(bs, device=x_src.device) < self.config.teacher_forcing_ratio).long()
			x_decoder[t] += teacher_forcing_mask * x_dst[:, t] + (1 - teacher_forcing_mask) * predicted_token
		return logits

