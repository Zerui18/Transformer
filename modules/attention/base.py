from torch import nn
from torch import Tensor
from typing import Tuple
import abc

class MultiHeadSelfAttentionBase(nn.Module, abc.ABC):

	''' Multi-head self attention.
	Implements a somewhat optimized version of the self attention by combining the q, k, v projections.
	
	Inputs:
		`x`: Tensor<Float>[B, T, C] input tensor.
		`tok_mask`: Tensor<Bool>[B, T] per-token mask applied to the `x`, false is masked out, true is preserved - masks both keys and queries.

	Outputs:
		Tensor<Float>[B, T, C] output tensor.
	'''

	def __init__(self, n_heads: int, emb_dim: int, dropout: float, bias: bool = False, is_causal: bool = False, output_attention: bool = False, tag: str = ''):
		super().__init__()
		self.is_causal = is_causal
		self.n_heads = n_heads
		self.emb_dim = emb_dim
		self.output_attention = output_attention
		self.tag = tag
		self.dropout = dropout
		self.bias = bias

	@abc.abstractmethod
	def _forward(self, x: Tensor, tok_mask: Tensor) -> Tuple[Tensor, Tensor]:
		pass

	def forward(self, x: Tensor, tok_mask: Tensor):
		output, att_weights = self._forward(x, tok_mask)
		if self.output_attention:
			return output, att_weights.detach().clone()
		else:
			return (output,)

class MultiHeadCrossAttentionBase(nn.Module, abc.ABC):

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

	def __init__(self, n_heads: int, emb_dim: int, dropout: float, bias: bool = False, output_attention: bool = False, tag: str = ''):
		super().__init__()
		self.n_heads = n_heads
		self.emb_dim = emb_dim
		self.output_attention = output_attention
		self.tag = tag
		self.dropout = dropout
		self.bias = bias

	@abc.abstractmethod
	def _forward(self, x_q: Tensor, x_kv: Tensor, q_tok_mask: Tensor, kv_tok_mask: Tensor) -> Tuple[Tensor, Tensor]:
		pass
			
	def forward(self, x_q: Tensor, x_kv: Tensor, q_tok_mask: Tensor, kv_tok_mask: Tensor):
		output, att_weights = self._forward(x_q, x_kv, q_tok_mask, kv_tok_mask)
		if self.output_attention:
			return output, att_weights.detach().clone()
		else:
			return (output,)