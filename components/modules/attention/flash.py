from torch.nn.attention import SDPBackend

from components.modules.attention.stock import StockSelfAttention, StockCrossAttention


class FlashSelfAttention(StockSelfAttention):
	'''Multi-head self-attention using the FlashAttention SDPA backend.

	Thin wrapper that overrides get_sdp_backends() to select the flash backend.
	'''

	def get_sdp_backends(self) -> list[SDPBackend]:
		'''Return SDPA backends with flash attention enabled.

		Returns:
			``list[SDPBackend]``: flash attention backend.
		'''
		return [SDPBackend.FLASH_ATTENTION]


class FlashCrossAttention(StockCrossAttention):
	'''Multi-head cross-attention using the FlashAttention SDPA backend.

	Thin wrapper that overrides get_sdp_backends() to select the flash backend.
	'''

	def get_sdp_backends(self) -> list[SDPBackend]:
		'''Return SDPA backends with flash attention enabled.

		Returns:
			``list[SDPBackend]``: flash attention backend.
		'''
		return [SDPBackend.FLASH_ATTENTION]

# Module-level aliases for lookup by transformer blocks
SELF_ATTENTION_CLS = FlashSelfAttention
CROSS_ATTENTION_CLS = FlashCrossAttention
