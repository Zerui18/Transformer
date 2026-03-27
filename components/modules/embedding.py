import torch
from torch import nn
from torch import Tensor


class PositionalEmbedding(nn.Module):
	''' Sinusoidal positional embedding table, precomputed and stored as a non-learnable buffer.

	Properties:
		1. encoding: Tensor  [float32, (max_len + 2, D)] precomputed sinusoidal embeddings.
	'''

	def __init__(self, emb_dim: int, max_len: int):
		''' Initialize the positional embedding table.

		Args:
			1. emb_dim: int  embedding dimension D.
			2. max_len: int  maximum sequence length supported.
		'''
		super().__init__()
		encoding = torch.zeros(max_len + 2, emb_dim, requires_grad=False)  # (max_len + 2, D)
		pos = torch.arange(0.0, max_len + 2, dtype=torch.float).unsqueeze(dim=1)  # (max_len + 2, 1)
		_2i = torch.arange(0, emb_dim, step=2).float()  # (D/2,)

		encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / emb_dim)))
		encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / emb_dim)))
		self.register_buffer('encoding', encoding)

	def forward(self, seq_len: int) -> Tensor:
		''' Return positional embeddings for the given sequence length.

		Args:
			1. seq_len: int  number of positions to retrieve.
		Returns:
			embeddings: Tensor  [float32, (seq_len, D)] positional embeddings.
		'''
		return self.encoding[:seq_len]


class PosNTokEmbedding(nn.Module):
	''' Combine learnable token embeddings with sinusoidal position embeddings.

	Properties:
		1. token_embedding_table: nn.Embedding  [float32, (vocab_size, D)] learnable token embeddings.
		2. position_embedding_table: PositionalEmbedding  sinusoidal positional embeddings.
		3. max_len: int  maximum sequence length.
	'''

	def __init__(self, vocab_size: int, emb_dim: int, max_len: int):
		''' Initialize token + position embedding.

		Args:
			1. vocab_size: int  number of tokens in the vocabulary.
			2. emb_dim: int  embedding dimension D.
			3. max_len: int  maximum sequence length.
		'''
		super().__init__()
		self.token_embedding_table = nn.Embedding(vocab_size, emb_dim)
		self.position_embedding_table = PositionalEmbedding(emb_dim, max_len)
		self.max_len = max_len

	def forward(self, x: Tensor) -> Tensor:
		''' Embed input token ids with combined token + positional embeddings.

		Args:
			1. x: Tensor  [int64, (B, T)] input token indices.
		Returns:
			embeddings: Tensor  [float32, (B, T, D)] token + positional embeddings.
		'''
		tok_embd = self.token_embedding_table(x)  # (B, T, D)
		pos_embd = self.position_embedding_table(x.size(1))  # (T, D)
		return tok_embd + pos_embd
