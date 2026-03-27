from typing import Any
from importlib import import_module

from torch import nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint


class TransformerFeedForward(nn.Module):
	''' Position-wise feed-forward network with GELU activation and optional dropout.

	Properties:
		1. net: nn.Sequential  linear -> GELU -> linear [-> dropout].
	'''

	def __init__(self, emb_dim: int, dropout: float, expansion_factor: int = 4):
		''' Initialize the feed-forward network.

		Args:
			1. emb_dim: int  input and output embedding dimension D.
			2. dropout: float  dropout rate applied after the second linear layer. 0 disables dropout.
			3. expansion_factor: int  hidden layer size multiplier (hidden = emb_dim * expansion_factor).
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
			1. x: Tensor  [float32, (B, T, D)] input features.
		Returns:
			out: Tensor  [float32, (B, T, D)] transformed features.
		'''
		return self.net(x)


class TransformerEncoderBlock(nn.Module):
	''' Single transformer encoder block with pre-norm self-attention and feed-forward.

	Properties:
		1. sa_module: nn.Module  multi-head self-attention module.
		2. fw_module: TransformerFeedForward  feed-forward network.
		3. ln1: nn.LayerNorm  pre-norm before self-attention.
		4. ln2: nn.LayerNorm  pre-norm before feed-forward.
	'''

	def __init__(self, idx: int, n_heads: int, emb_dim: int, dropout: float,
				 bias: bool = False, attention_type: str = 'vanilla',
				 output_attention: bool = False):
		''' Initialize a transformer encoder block.

		Args:
			1. idx: int  block index (used for attention tagging).
			2. n_heads: int  number of attention heads H.
			3. emb_dim: int  embedding dimension D.
			4. dropout: float  dropout rate.
			5. bias: bool  whether to use bias in attention projections.
			6. attention_type: str  attention module name under components.modules.attention.
			7. output_attention: bool  whether to output attention weights.
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
			1. src: Tensor  [float32, (B, T, D)] source embeddings.
			2. src_mask: Tensor  [bool, (B, T)] source token mask (True = keep).
		Returns:
			out: Tensor  [float32, (B, T, D)] encoded features.
		'''
		# self-attention with pre-norm and residual
		x = src + self.sa_module(self.ln1(src), src_mask)[0]
		# feed-forward with pre-norm and residual
		x = x + self.fw_module(self.ln2(x))
		return x


class TransformerEncoder(nn.Module):
	''' Stack of transformer encoder blocks with optional gradient checkpointing.

	Properties:
		1. blocks: nn.ModuleList  list of TransformerEncoderBlock.
		2. use_grad_ckpt: bool  whether to use gradient checkpointing.
	'''

	def __init__(self, n_blocks: int, n_heads: int, emb_dim: int, dropout: float,
				 bias: bool = False, use_grad_ckpt: bool = False,
				 attention_type: str = 'vanilla', output_attention: bool = False):
		''' Initialize the transformer encoder.

		Args:
			1. n_blocks: int  number of encoder blocks.
			2. n_heads: int  number of attention heads H.
			3. emb_dim: int  embedding dimension D.
			4. dropout: float  dropout rate.
			5. bias: bool  whether to use bias in attention projections.
			6. use_grad_ckpt: bool  enable gradient checkpointing to save memory.
			7. attention_type: str  attention module name.
			8. output_attention: bool  whether to output attention weights.
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
			1. src: Tensor  [float32, (B, T, D)] source embeddings.
			2. src_mask: Tensor  [bool, (B, T)] source token mask.
		Returns:
			out: Tensor  [float32, (B, T, D)] encoded features.
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

	Properties:
		1. sa_module: nn.Module  masked multi-head self-attention module.
		2. ca_module: nn.Module  multi-head cross-attention module.
		3. fw_module: TransformerFeedForward  feed-forward network.
		4. ln1: nn.LayerNorm  pre-norm before self-attention.
		5. ln2: nn.LayerNorm  pre-norm before cross-attention (query path).
		6. ln3: nn.LayerNorm  pre-norm before feed-forward.
		7. ln_enc: nn.LayerNorm  pre-norm for encoder output (key-value path in cross-attention).
	'''

	def __init__(self, idx: int, n_heads: int, emb_dim: int, dropout: float,
				 bias: bool = False, attention_type: str = 'vanilla',
				 output_attention: bool = False):
		''' Initialize a transformer decoder block.

		Args:
			1. idx: int  block index.
			2. n_heads: int  number of attention heads H.
			3. emb_dim: int  embedding dimension D.
			4. dropout: float  dropout rate.
			5. bias: bool  whether to use bias in attention projections.
			6. attention_type: str  attention module name.
			7. output_attention: bool  whether to output attention weights.
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
			1. src: Tensor  [float32, (B, Ts, D)] encoder output.
			2. tgt: Tensor  [float32, (B, Tt, D)] target embeddings.
			3. src_mask: Tensor  [bool, (B, Ts)] source token mask.
			4. tgt_mask: Tensor  [bool, (B, Tt)] target token mask.
		Returns:
			out: Tensor  [float32, (B, Tt, D)] decoded features.
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

	Properties:
		1. blocks: nn.ModuleList  list of TransformerDecoderBlock.
		2. use_grad_ckpt: bool  whether to use gradient checkpointing.
	'''

	def __init__(self, n_blocks: int, n_heads: int, emb_dim: int, dropout: float,
				 bias: bool = False, use_grad_ckpt: bool = False,
				 attention_type: str = 'vanilla', output_attention: bool = False):
		''' Initialize the transformer decoder.

		Args:
			1. n_blocks: int  number of decoder blocks.
			2. n_heads: int  number of attention heads H.
			3. emb_dim: int  embedding dimension D.
			4. dropout: float  dropout rate.
			5. bias: bool  whether to use bias in attention projections.
			6. use_grad_ckpt: bool  enable gradient checkpointing.
			7. attention_type: str  attention module name.
			8. output_attention: bool  whether to output attention weights.
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
			1. src: Tensor  [float32, (B, Ts, D)] encoder output.
			2. tgt: Tensor  [float32, (B, Tt, D)] target embeddings.
			3. src_mask: Tensor  [bool, (B, Ts)] source token mask.
			4. tgt_mask: Tensor  [bool, (B, Tt)] target token mask.
		Returns:
			out: Tensor  [float32, (B, Tt, D)] decoded features.
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

	Properties:
		1. ln: nn.LayerNorm  final layer normalization.
		2. logits_head: nn.Linear  [float32, (D, V)] projects embeddings to vocab logits.
	'''

	def __init__(self, emb_dim: int, tgt_vocab_size: int):
		''' Initialize the LM head.

		Args:
			1. emb_dim: int  embedding dimension D.
			2. tgt_vocab_size: int  target vocabulary size V.
		'''
		super().__init__()
		self.ln = nn.LayerNorm(emb_dim)
		self.logits_head = nn.Linear(emb_dim, tgt_vocab_size, bias=False)

	def forward(self, x: Tensor) -> Tensor:
		''' Project embeddings to vocabulary logits.

		Args:
			1. x: Tensor  [float32, (B, T, D)] decoder output embeddings.
		Returns:
			logits: Tensor  [float32, (B, T, V)] unnormalized logits over vocabulary.
		'''
		return self.logits_head(self.ln(x))
