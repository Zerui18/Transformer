from typing import Any
from importlib import import_module

from torch import nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint


class TransformerFeedForward(nn.Module):
	''' Position-wise feed-forward network with GELU activation and optional dropout.

	Attributes:
		net: ``nn.Sequential``: linear -> GELU -> linear [-> dropout].
	'''

	def __init__(self, emb_dim: int, dropout: float, expansion_factor: int = 4):
		''' Initialize the feed-forward network.

		Args:
			emb_dim: ``int``: input and output embedding dimension D.
			dropout: ``float``: dropout rate applied after the second linear layer. 0 disables dropout.
			expansion_factor: ``int``: hidden layer size multiplier (hidden = emb_dim * expansion_factor).
		'''
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(emb_dim, expansion_factor * emb_dim),
			nn.GELU(approximate='tanh'),
			nn.Linear(expansion_factor * emb_dim, emb_dim),
		)
		if dropout:
			self.net.append(nn.Dropout(dropout))

	def forward(self, x: Tensor) -> Tensor:
		''' Apply position-wise feed-forward transformation.

		Args:
			x: ``Tensor[(B, T, D), float32]``: input features.
		Returns:
			``Tensor[(B, T, D), float32]``: transformed features.
		'''
		return self.net(x)


class TransformerEncoderBlock(nn.Module):
	''' Single transformer encoder block with pre-norm self-attention and feed-forward.

	Attributes:
		sa_module: ``nn.Module``: multi-head self-attention module.
		fw_module: ``TransformerFeedForward``: feed-forward network.
		ln1: ``nn.LayerNorm``: pre-norm before self-attention.
		ln2: ``nn.LayerNorm``: pre-norm before feed-forward.
	'''

	def __init__(self, idx: int, n_heads: int, emb_dim: int, dropout: float,
				 bias: bool = False, attention_type: str = 'vanilla',
				 output_attention: bool = False):
		''' Initialize a transformer encoder block.

		Args:
			idx: ``int``: block index (used for attention tagging).
			n_heads: ``int``: number of attention heads H.
			emb_dim: ``int``: embedding dimension D.
			dropout: ``float``: dropout rate.
			bias: ``bool``: whether to use bias in attention projections.
			attention_type: ``str``: attention module name under components.modules.attention.
			output_attention: ``bool``: whether to output attention weights.
		'''
		super().__init__()
		self.output_attention = output_attention
		# look up attention class from the attention subpackage
		attn_module = import_module(f'.attention.{attention_type}', 'components.modules')
		sa_class = attn_module.SELF_ATTENTION_CLS
		self.sa_module = sa_class(
			n_heads, emb_dim, dropout, bias,
			is_causal=False, output_attention=output_attention,
			tag=f'encoder_sa_{idx}')
		self.fw_module = TransformerFeedForward(emb_dim, dropout)
		self.ln1 = nn.LayerNorm(emb_dim)
		self.ln2 = nn.LayerNorm(emb_dim)

	def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
		''' Apply encoder block: pre-norm self-attention + residual, then pre-norm FFN + residual.

		Args:
			src: ``Tensor[(B, T, D), float32]``: source embeddings.
			src_mask: ``Tensor[(B, T), bool]``: source token mask (True = keep).
		Returns:
			``Tensor[(B, T, D), float32]``: encoded features.
		'''
		# self-attention with pre-norm and residual
		x = src + self.sa_module(self.ln1(src), src_mask)[0]
		# feed-forward with pre-norm and residual
		x = x + self.fw_module(self.ln2(x))
		return x


class TransformerEncoder(nn.Module):
	''' Stack of transformer encoder blocks with optional gradient checkpointing.

	Attributes:
		blocks: ``nn.ModuleList``: list of TransformerEncoderBlock.
		use_grad_ckpt: ``bool``: whether to use gradient checkpointing.
	'''

	def __init__(self, n_blocks: int, n_heads: int, emb_dim: int, dropout: float,
				 bias: bool = False, use_grad_ckpt: bool = False,
				 attention_type: str = 'vanilla', output_attention: bool = False):
		''' Initialize the transformer encoder.

		Args:
			n_blocks: ``int``: number of encoder blocks.
			n_heads: ``int``: number of attention heads H.
			emb_dim: ``int``: embedding dimension D.
			dropout: ``float``: dropout rate.
			bias: ``bool``: whether to use bias in attention projections.
			use_grad_ckpt: ``bool``: enable gradient checkpointing to save memory.
			attention_type: ``str``: attention module name.
			output_attention: ``bool``: whether to output attention weights.
		'''
		super().__init__()
		self.blocks = nn.ModuleList([
			TransformerEncoderBlock(idx, n_heads, emb_dim, dropout, bias,
								   attention_type, output_attention)
			for idx in range(n_blocks)
		])
		self.use_grad_ckpt = use_grad_ckpt

	def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
		''' Pass source through all encoder blocks.

		Args:
			src: ``Tensor[(B, T, D), float32]``: source embeddings.
			src_mask: ``Tensor[(B, T), bool]``: source token mask.
		Returns:
			``Tensor[(B, T, D), float32]``: encoded features.
		'''
		x = src
		for block in self.blocks:
			if self.use_grad_ckpt:
				forward = lambda *inputs, _block=block: _block(*inputs)
				x = checkpoint(forward, x, src_mask, use_reentrant=False)
			else:
				x = block(x, src_mask)
		return x


class TransformerDecoderBlock(nn.Module):
	''' Single transformer decoder block with pre-norm self-attention, cross-attention, and feed-forward.

	Attributes:
		sa_module: ``nn.Module``: masked multi-head self-attention module.
		ca_module: ``nn.Module``: multi-head cross-attention module.
		fw_module: ``TransformerFeedForward``: feed-forward network.
		ln1: ``nn.LayerNorm``: pre-norm before self-attention.
		ln2: ``nn.LayerNorm``: pre-norm before cross-attention (query path).
		ln3: ``nn.LayerNorm``: pre-norm before feed-forward.
		ln_enc: ``nn.LayerNorm``: pre-norm for encoder output (key-value path in cross-attention).
	'''

	def __init__(self, idx: int, n_heads: int, emb_dim: int, dropout: float,
				 bias: bool = False, attention_type: str = 'vanilla',
				 output_attention: bool = False):
		''' Initialize a transformer decoder block.

		Args:
			idx: ``int``: block index.
			n_heads: ``int``: number of attention heads H.
			emb_dim: ``int``: embedding dimension D.
			dropout: ``float``: dropout rate.
			bias: ``bool``: whether to use bias in attention projections.
			attention_type: ``str``: attention module name.
			output_attention: ``bool``: whether to output attention weights.
		'''
		super().__init__()
		self.output_attention = output_attention
		attn_module = import_module(f'.attention.{attention_type}', 'components.modules')
		sa_class = attn_module.SELF_ATTENTION_CLS
		ca_class = attn_module.CROSS_ATTENTION_CLS
		self.sa_module = sa_class(
			n_heads, emb_dim, dropout, bias,
			is_causal=True, output_attention=output_attention,
			tag=f'decoder_sa_{idx}')
		self.ca_module = ca_class(
			n_heads, emb_dim, dropout, bias,
			output_attention=output_attention, tag=f'decoder_ca_{idx}')
		self.fw_module = TransformerFeedForward(emb_dim, dropout)
		self.ln1 = nn.LayerNorm(emb_dim)
		self.ln2 = nn.LayerNorm(emb_dim)
		self.ln3 = nn.LayerNorm(emb_dim)
		# separate LayerNorm for encoder output in cross-attention (fixes shared-ln2 bug)
		self.ln_enc = nn.LayerNorm(emb_dim)

	def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor, tgt_mask: Tensor) -> Tensor:
		''' Apply decoder block: self-attn → cross-attn → FFN, each with pre-norm + residual.

		Args:
			src: ``Tensor[(B, Ts, D), float32]``: encoder output.
			tgt: ``Tensor[(B, Tt, D), float32]``: target embeddings.
			src_mask: ``Tensor[(B, Ts), bool]``: source token mask.
			tgt_mask: ``Tensor[(B, Tt), bool]``: target token mask.
		Returns:
			``Tensor[(B, Tt, D), float32]``: decoded features.
		'''
		# causal self-attention with pre-norm and residual
		x = tgt + self.sa_module(self.ln1(tgt), tgt_mask)[0]
		# cross-attention with separate pre-norms for query and key-value paths
		x = x + self.ca_module(self.ln2(x), self.ln_enc(src), tgt_mask, src_mask)[0]
		# feed-forward with pre-norm and residual
		x = x + self.fw_module(self.ln3(x))
		return x


class TransformerDecoder(nn.Module):
	''' Stack of transformer decoder blocks with optional gradient checkpointing.

	Attributes:
		blocks: ``nn.ModuleList``: list of TransformerDecoderBlock.
		use_grad_ckpt: ``bool``: whether to use gradient checkpointing.
	'''

	def __init__(self, n_blocks: int, n_heads: int, emb_dim: int, dropout: float,
				 bias: bool = False, use_grad_ckpt: bool = False,
				 attention_type: str = 'vanilla', output_attention: bool = False):
		''' Initialize the transformer decoder.

		Args:
			n_blocks: ``int``: number of decoder blocks.
			n_heads: ``int``: number of attention heads H.
			emb_dim: ``int``: embedding dimension D.
			dropout: ``float``: dropout rate.
			bias: ``bool``: whether to use bias in attention projections.
			use_grad_ckpt: ``bool``: enable gradient checkpointing.
			attention_type: ``str``: attention module name.
			output_attention: ``bool``: whether to output attention weights.
		'''
		super().__init__()
		self.blocks = nn.ModuleList([
			TransformerDecoderBlock(idx, n_heads, emb_dim, dropout, bias,
								   attention_type, output_attention)
			for idx in range(n_blocks)
		])
		self.use_grad_ckpt = use_grad_ckpt

	def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor, tgt_mask: Tensor) -> Tensor:
		''' Pass target through all decoder blocks.

		Args:
			src: ``Tensor[(B, Ts, D), float32]``: encoder output.
			tgt: ``Tensor[(B, Tt, D), float32]``: target embeddings.
			src_mask: ``Tensor[(B, Ts), bool]``: source token mask.
			tgt_mask: ``Tensor[(B, Tt), bool]``: target token mask.
		Returns:
			``Tensor[(B, Tt, D), float32]``: decoded features.
		'''
		x = tgt
		for block in self.blocks:
			if self.use_grad_ckpt:
				forward = lambda *inputs, _block=block: _block(*inputs)
				x = checkpoint(forward, src, x, src_mask, tgt_mask, use_reentrant=False)
			else:
				x = block(src, x, src_mask, tgt_mask)
		return x


class TransformerLMHead(nn.Module):
	''' Language model head: LayerNorm followed by a linear projection to vocabulary logits.

	Attributes:
		ln: ``nn.LayerNorm``: final layer normalization.
		logits_head: ``nn.Linear``: [float32, (D, V)] projects embeddings to vocab logits.
	'''

	def __init__(self, emb_dim: int, tgt_vocab_size: int):
		''' Initialize the LM head.

		Args:
			emb_dim: ``int``: embedding dimension D.
			tgt_vocab_size: ``int``: target vocabulary size V.
		'''
		super().__init__()
		self.ln = nn.LayerNorm(emb_dim)
		self.logits_head = nn.Linear(emb_dim, tgt_vocab_size, bias=False)

	def forward(self, x: Tensor) -> Tensor:
		''' Project embeddings to vocabulary logits.

		Args:
			x: ``Tensor[(B, T, D), float32]``: decoder output embeddings.
		Returns:
			``Tensor[(B, T, V), float32]``: unnormalized logits over vocabulary.
		'''
		return self.logits_head(self.ln(x))
