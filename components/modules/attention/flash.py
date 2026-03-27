from components.modules.attention.stock import StockSelfAttention, StockCrossAttention


class FlashSelfAttention(StockSelfAttention):
	'''Multi-head self-attention using the FlashAttention SDPA backend.

	Properties:
		(inherited from StockSelfAttention)

	Thin wrapper around StockSelfAttention that overrides get_attention_args() to
	enable the flash attention backend and disable the math backend. All other
	behaviour, including the inability to output attention weights, is inherited.
	'''

	def get_attention_args(self) -> dict[str, bool]:
		'''Return SDPA backend flags with flash attention enabled.

		Returns:
			args: dict[str, bool]  keys are enable_math, enable_flash, enable_mem_efficient.
		'''
		return {
			'enable_math': False,
			'enable_flash': True,
			'enable_mem_efficient': False,
		}


class FlashCrossAttention(StockCrossAttention):
	'''Multi-head cross-attention using the FlashAttention SDPA backend.

	Properties:
		(inherited from StockCrossAttention)

	Thin wrapper around StockCrossAttention that overrides get_attention_args() to
	enable the flash attention backend and disable the math backend. All other
	behaviour, including the inability to output attention weights, is inherited.
	'''

	def get_attention_args(self) -> dict[str, bool]:
		'''Return SDPA backend flags with flash attention enabled.

		Returns:
			args: dict[str, bool]  keys are enable_math, enable_flash, enable_mem_efficient.
		'''
		return {
			'enable_math': False,
			'enable_flash': True,
			'enable_mem_efficient': False,
		}

# Module-level aliases for lookup by transformer blocks
SELF_ATTENTION_CLS = FlashSelfAttention
CROSS_ATTENTION_CLS = FlashCrossAttention
