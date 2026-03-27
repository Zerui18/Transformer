from torch.nn.attention import SDPBackend

from components.modules.attention.stock import StockSelfAttention, StockCrossAttention


class MemEfficientSelfAttention(StockSelfAttention):
	'''Multi-head self-attention using the memory-efficient SDPA backend.

	Thin wrapper that overrides get_sdp_backends() to select the mem-efficient backend.
	'''

	def get_sdp_backends(self) -> list[SDPBackend]:
		'''Return SDPA backends with memory-efficient attention enabled.

		Returns:
			``list[SDPBackend]``: memory-efficient backend.
		'''
		return [SDPBackend.EFFICIENT_ATTENTION]


class MemEfficientCrossAttention(StockCrossAttention):
	'''Multi-head cross-attention using the memory-efficient SDPA backend.

	Thin wrapper that overrides get_sdp_backends() to select the mem-efficient backend.
	'''

	def get_sdp_backends(self) -> list[SDPBackend]:
		'''Return SDPA backends with memory-efficient attention enabled.

		Returns:
			``list[SDPBackend]``: memory-efficient backend.
		'''
		return [SDPBackend.EFFICIENT_ATTENTION]

# Module-level aliases for lookup by transformer blocks
SELF_ATTENTION_CLS = MemEfficientSelfAttention
CROSS_ATTENTION_CLS = MemEfficientCrossAttention
