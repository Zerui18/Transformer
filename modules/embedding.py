import torch
from torch import Tensor
from torch import nn

class PositionalEmbedding(nn.Module):
	''' Sinosuidal positional embedding. '''

	def __init__(self, emb_dim: int, max_len: int):
		super().__init__()

		encoding = torch.zeros(max_len + 2, emb_dim, requires_grad=False)
		pos = torch.arange(0.0, max_len + 2, dtype=torch.float).unsqueeze(dim=1)
		_2i = torch.arange(0, emb_dim, step=2).float()

		encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / emb_dim)))
		encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / emb_dim)))

		self.register_buffer('encoding', encoding)

	def forward(self, x: Tensor):
		return self.encoding[:x]

class PosNTokEmbedding(nn.Module):
	''' Apply learnable token and sinosuidal position embeddings to input tokens. '''

	def __init__(self, vocab_size: int, emb_dim: int, max_len: int):
		super().__init__()
		self.token_embedding_table = nn.Embedding(vocab_size, emb_dim)
		self.position_embedding_table = PositionalEmbedding(emb_dim, max_len)
		self.max_len = max_len
	
	def forward(self, x: Tensor):
		tok_embd = self.token_embedding_table(x)
		pos_embd = self.position_embedding_table(torch.tensor(x.size(1), dtype=torch.long, device=x.device))
		return tok_embd + pos_embd