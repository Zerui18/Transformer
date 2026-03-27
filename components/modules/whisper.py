from torch import nn
from torch import Tensor
from torch.nn import functional as F


class AudioEncoder(nn.Module):
	''' CNN-based audio encoder that converts mel spectrograms into embeddings.

	Applies a stack of 1D convolutions with GELU activations. The final layer
	uses stride 2 to halve the temporal dimension.

	Properties:
		1. n_layers: int  number of convolutional layers.
		2. kernel_size: int  kernel size for all convolutions.
		3. n_filters: int  number of output channels (= embedding dim).
		4. convs: nn.ModuleList  the convolutional layers.
	'''

	def __init__(self, n_layers: int = 2, kernel_size: int = 3, n_filters: int = 256, n_mels: int = 80):
		''' Initialize the audio encoder.

		Args:
			1. n_layers: int  number of 1D convolution layers.
			2. kernel_size: int  convolution kernel size.
			3. n_filters: int  number of output channels (embedding dimension D).
			4. n_mels: int  number of mel frequency bins in the input.
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
			1. x: Tensor  [float32, (B, T, M)] mel spectrogram input, M = mel bins.
		Returns:
			out: Tensor  [float32, (B, T//2, D)] encoded features, D = n_filters.
		'''
		x = x.transpose(-2, -1)  # (B, M, T) — channels first for Conv1d
		for conv in self.convs:
			x = F.gelu(conv(x))
		return x.transpose(-2, -1)  # (B, T//2, D) — channels last
