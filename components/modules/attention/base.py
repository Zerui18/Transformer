import abc

from torch import nn, Tensor


class MultiHeadSelfAttentionBase(nn.Module, abc.ABC):
	'''Abstract base class for multi-head self-attention modules.

	Properties:
		1. n_heads: int  number of attention heads.
		2. emb_dim: int  embedding dimension D, must be divisible by n_heads.
		3. dropout: float  dropout probability applied to attention weights and residual.
		4. bias: bool  whether to use bias in linear projections.
		5. is_causal: bool  whether to apply a causal (lower-triangular) mask.
		6. output_attention: bool  whether forward() returns attention weights alongside output.
		7. tag: str  optional identifier tag for this attention instance.

	Subclasses must implement _forward() which returns (output, att_weights) where att_weights
	may be None if the backend does not support weight extraction. The public forward() method
	handles the output_attention logic by either returning (output,) or (output, att_weights).
	'''

	def __init__(
		self,
		n_heads: int,
		emb_dim: int,
		dropout: float,
		bias: bool = False,
		is_causal: bool = False,
		output_attention: bool = False,
		tag: str = '',
	) -> None:
		'''Initialise multi-head self-attention base.

		Args:
			1. n_heads: int  number of attention heads.
			2. emb_dim: int  embedding dimension, must be divisible by n_heads.
			3. dropout: float  dropout probability.
			4. bias: bool  whether linear projections include bias terms.
			5. is_causal: bool  whether to apply causal masking.
			6. output_attention: bool  whether to return attention weights from forward().
			7. tag: str  optional identifier tag.
		'''
		super().__init__()
		self.is_causal = is_causal
		self.n_heads = n_heads
		self.emb_dim = emb_dim
		self.output_attention = output_attention
		self.tag = tag
		self.dropout = dropout
		self.bias = bias

	@abc.abstractmethod
	def _forward(self, x: Tensor, tok_mask: Tensor) -> tuple[Tensor, Tensor | None]:
		'''Compute self-attention (implemented by subclasses).

		Args:
			1. x: Tensor  [float32, (B, T, D)] input embeddings.
			2. tok_mask: Tensor  [bool, (B, T)] per-token mask; False is masked out, True is preserved.
		Returns:
			result: tuple[Tensor, Tensor | None]  (output (B, T, D), attention_weights (B, H, T, T) or None).
		'''
		pass

	def forward(self, x: Tensor, tok_mask: Tensor) -> tuple[Tensor, ...]:
		'''Run self-attention and optionally return attention weights.

		Args:
			1. x: Tensor  [float32, (B, T, D)] input embeddings.
			2. tok_mask: Tensor  [bool, (B, T)] per-token mask; False is masked out, True is preserved.
		Returns:
			result: tuple[Tensor, ...]  (output,) or (output, att_weights) depending on output_attention.
		'''
		output, att_weights = self._forward(x, tok_mask)
		if self.output_attention:
			return output, att_weights.detach().clone()
		else:
			return (output,)


class MultiHeadCrossAttentionBase(nn.Module, abc.ABC):
	'''Abstract base class for multi-head cross-attention modules.

	Properties:
		1. n_heads: int  number of attention heads.
		2. emb_dim: int  embedding dimension D, must be divisible by n_heads.
		3. dropout: float  dropout probability applied to attention weights and residual.
		4. bias: bool  whether to use bias in linear projections.
		5. output_attention: bool  whether forward() returns attention weights alongside output.
		6. tag: str  optional identifier tag for this attention instance.

	Subclasses must implement _forward() which returns (output, att_weights) where att_weights
	may be None if the backend does not support weight extraction. The public forward() method
	handles the output_attention logic by either returning (output,) or (output, att_weights).
	'''

	def __init__(
		self,
		n_heads: int,
		emb_dim: int,
		dropout: float,
		bias: bool = False,
		output_attention: bool = False,
		tag: str = '',
	) -> None:
		'''Initialise multi-head cross-attention base.

		Args:
			1. n_heads: int  number of attention heads.
			2. emb_dim: int  embedding dimension, must be divisible by n_heads.
			3. dropout: float  dropout probability.
			4. bias: bool  whether linear projections include bias terms.
			5. output_attention: bool  whether to return attention weights from forward().
			6. tag: str  optional identifier tag.
		'''
		super().__init__()
		self.n_heads = n_heads
		self.emb_dim = emb_dim
		self.output_attention = output_attention
		self.tag = tag
		self.dropout = dropout
		self.bias = bias

	@abc.abstractmethod
	def _forward(self, x_q: Tensor, x_kv: Tensor, q_tok_mask: Tensor, kv_tok_mask: Tensor) -> tuple[Tensor, Tensor | None]:
		'''Compute cross-attention (implemented by subclasses).

		Args:
			1. x_q: Tensor  [float32, (B, Tq, D)] query embeddings.
			2. x_kv: Tensor  [float32, (B, Tk, D)] key/value embeddings.
			3. q_tok_mask: Tensor  [bool, (B, Tq)] query token mask; False is masked out.
			4. kv_tok_mask: Tensor  [bool, (B, Tk)] key/value token mask; False is masked out.
		Returns:
			result: tuple[Tensor, Tensor | None]  (output (B, Tq, D), attention_weights (B, H, Tq, Tk) or None).
		'''
		pass

	def forward(self, x_q: Tensor, x_kv: Tensor, q_tok_mask: Tensor, kv_tok_mask: Tensor) -> tuple[Tensor, ...]:
		'''Run cross-attention and optionally return attention weights.

		Args:
			1. x_q: Tensor  [float32, (B, Tq, D)] query embeddings.
			2. x_kv: Tensor  [float32, (B, Tk, D)] key/value embeddings.
			3. q_tok_mask: Tensor  [bool, (B, Tq)] query token mask; False is masked out.
			4. kv_tok_mask: Tensor  [bool, (B, Tk)] key/value token mask; False is masked out.
		Returns:
			result: tuple[Tensor, ...]  (output,) or (output, att_weights) depending on output_attention.
		'''
		output, att_weights = self._forward(x_q, x_kv, q_tok_mask, kv_tok_mask)
		if self.output_attention:
			return output, att_weights.detach().clone()
		else:
			return (output,)
