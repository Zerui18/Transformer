from components.modules.attention.stock import StockSelfAttention, StockCrossAttention


class MemEfficientSelfAttention(StockSelfAttention):
	'''Multi-head self-attention using the memory-efficient SDPA backend.

	Attributes:
		(inherited from StockSelfAttention)

	Thin wrapper around StockSelfAttention that overrides get_attention_args() to
	enable the memory-efficient attention backend and disable the math backend. All
	other behaviour, including the inability to output attention weights, is inherited.
	'''

	def get_attention_args(self) -> dict[str, bool]:
		'''Return SDPA backend flags with memory-efficient attention enabled.

		Returns:
			``dict[str, bool]``: keys are enable_math, enable_flash, enable_mem_efficient.
		'''
		return {
			'enable_math': False,
			'enable_flash': False,
			'enable_mem_efficient': True,
		}


class MemEfficientCrossAttention(StockCrossAttention):
	'''Multi-head cross-attention using the memory-efficient SDPA backend.

	Attributes:
		(inherited from StockCrossAttention)

	Thin wrapper around StockCrossAttention that overrides get_attention_args() to
	enable the memory-efficient attention backend and disable the math backend. All
	other behaviour, including the inability to output attention weights, is inherited.
	'''

	def get_attention_args(self) -> dict[str, bool]:
		'''Return SDPA backend flags with memory-efficient attention enabled.

		Returns:
			``dict[str, bool]``: keys are enable_math, enable_flash, enable_mem_efficient.
		'''
		return {
			'enable_math': False,
			'enable_flash': False,
			'enable_mem_efficient': True,
		}

# Module-level aliases for lookup by transformer blocks
SELF_ATTENTION_CLS = MemEfficientSelfAttention
CROSS_ATTENTION_CLS = MemEfficientCrossAttention
