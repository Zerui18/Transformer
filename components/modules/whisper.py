from torch import nn
from torch import Tensor
from torch.nn import functional as F


class AudioEncoder(nn.Module):
	''' CNN-based audio encoder that converts mel spectrograms into embeddings.

	Applies a stack of 1D convolutions with GELU activations. The final layer
	uses stride 2 to halve the temporal dimension.

	Attributes:
		n_layers: ``int``: number of convolutional layers.
		kernel_size: ``int``: kernel size for all convolutions.
		n_filters: ``int``: number of output channels (= embedding dim).
		convs: ``nn.ModuleList``: the convolutional layers.
	'''

	def __init__(self, n_layers: int = 2, kernel_size: int = 3, n_filters: int = 256, n_mels: int = 80):
		''' Initialize the audio encoder.

		Args:
			n_layers: ``int``: number of 1D convolution layers.
			kernel_size: ``int``: convolution kernel size.
			n_filters: ``int``: number of output channels (embedding dimension D).
			n_mels: ``int``: number of mel frequency bins in the input.
		'''
		super().__init__()
		self.n_layers = n_layers
		self.kernel_size = kernel_size
		self.n_filters = n_filters
		# final layer uses stride=2 to halve temporal dim
		self.convs = nn.ModuleList([
			nn.Conv1d(
				(n_filters if i > 0 else n_mels), n_filters,
				kernel_size, stride=(1 if i < n_layers - 1 else 2),
				padding=1, bias=False
			)
			for i in range(n_layers)
		])

	def forward(self, x: Tensor) -> Tensor:
		''' Encode mel spectrogram features into embeddings.

		Args:
			x: ``Tensor[(B, T, M), float32]``: mel spectrogram input, M = mel bins.
		Returns:
			``Tensor[(B, T//2, D), float32]``: encoded features, D = n_filters.
		'''
		x = x.transpose(-2, -1)  # (B, M, T) — channels first for Conv1d
		for conv in self.convs:
			x = F.gelu(conv(x))
		return x.transpose(-2, -1)  # (B, T//2, D) — channels last
