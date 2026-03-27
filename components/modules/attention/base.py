import abc

from torch import nn, Tensor


class MultiHeadSelfAttentionBase(nn.Module, abc.ABC):
	'''Abstract base class for multi-head self-attention modules.

	Attributes:
		n_heads: ``int``: number of attention heads.
		emb_dim: ``int``: embedding dimension D, must be divisible by n_heads.
		dropout: ``float``: dropout probability applied to attention weights and residual.
		bias: ``bool``: whether to use bias in linear projections.
		is_causal: ``bool``: whether to apply a causal (lower-triangular) mask.
		output_attention: ``bool``: whether forward() returns attention weights alongside output.
		tag: ``str``: optional identifier tag for this attention instance.

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
			n_heads: ``int``: number of attention heads.
			emb_dim: ``int``: embedding dimension, must be divisible by n_heads.
			dropout: ``float``: dropout probability.
			bias: ``bool``: whether linear projections include bias terms.
			is_causal: ``bool``: whether to apply causal masking.
			output_attention: ``bool``: whether to return attention weights from forward().
			tag: ``str``: optional identifier tag.
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
			x: ``Tensor[(B, T, D), float32]``: input embeddings.
			tok_mask: ``Tensor[(B, T), bool]``: per-token mask; False is masked out, True is preserved.
		Returns:
			``tuple[Tensor, Tensor | None]``: (output (B, T, D), attention_weights (B, H, T, T) or None).
		'''
		pass

	def forward(self, x: Tensor, tok_mask: Tensor) -> tuple[Tensor, ...]:
		'''Run self-attention and optionally return attention weights.

		Args:
			x: ``Tensor[(B, T, D), float32]``: input embeddings.
			tok_mask: ``Tensor[(B, T), bool]``: per-token mask; False is masked out, True is preserved.
		Returns:
			``tuple[Tensor, ...]``: (output,) or (output, att_weights) depending on output_attention.
		'''
		output, att_weights = self._forward(x, tok_mask)
		if self.output_attention:
			if att_weights is None:
				raise RuntimeError(
					f'{self.__class__.__name__}._forward() returned None for att_weights '
					'but output_attention=True. This backend does not support attention weight output.')
			return output, att_weights.detach().clone()
		else:
			return (output,)


class MultiHeadCrossAttentionBase(nn.Module, abc.ABC):
	'''Abstract base class for multi-head cross-attention modules.

	Attributes:
		n_heads: ``int``: number of attention heads.
		emb_dim: ``int``: embedding dimension D, must be divisible by n_heads.
		dropout: ``float``: dropout probability applied to attention weights and residual.
		bias: ``bool``: whether to use bias in linear projections.
		output_attention: ``bool``: whether forward() returns attention weights alongside output.
		tag: ``str``: optional identifier tag for this attention instance.

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
			n_heads: ``int``: number of attention heads.
			emb_dim: ``int``: embedding dimension, must be divisible by n_heads.
			dropout: ``float``: dropout probability.
			bias: ``bool``: whether linear projections include bias terms.
			output_attention: ``bool``: whether to return attention weights from forward().
			tag: ``str``: optional identifier tag.
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
			x_q: ``Tensor[(B, Tq, D), float32]``: query embeddings.
			x_kv: ``Tensor[(B, Tk, D), float32]``: key/value embeddings.
			q_tok_mask: ``Tensor[(B, Tq), bool]``: query token mask; False is masked out.
			kv_tok_mask: ``Tensor[(B, Tk), bool]``: key/value token mask; False is masked out.
		Returns:
			``tuple[Tensor, Tensor | None]``: (output (B, Tq, D), attention_weights (B, H, Tq, Tk) or None).
		'''
		pass

	def forward(self, x_q: Tensor, x_kv: Tensor, q_tok_mask: Tensor, kv_tok_mask: Tensor) -> tuple[Tensor, ...]:
		'''Run cross-attention and optionally return attention weights.

		Args:
			x_q: ``Tensor[(B, Tq, D), float32]``: query embeddings.
			x_kv: ``Tensor[(B, Tk, D), float32]``: key/value embeddings.
			q_tok_mask: ``Tensor[(B, Tq), bool]``: query token mask; False is masked out.
			kv_tok_mask: ``Tensor[(B, Tk), bool]``: key/value token mask; False is masked out.
		Returns:
			``tuple[Tensor, ...]``: (output,) or (output, att_weights) depending on output_attention.
		'''
		output, att_weights = self._forward(x_q, x_kv, q_tok_mask, kv_tok_mask)
		if self.output_attention:
			if att_weights is None:
				raise RuntimeError(
					f'{self.__class__.__name__}._forward() returned None for att_weights '
					'but output_attention=True. This backend does not support attention weight output.')
			return output, att_weights.detach().clone()
		else:
			return (output,)
