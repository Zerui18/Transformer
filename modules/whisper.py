import torch
from torch import nn
from torch import Tensor

class AudioEncoder(nn.Module):
    
	def __init__(self, n_layers: int = 2, kernel_size: int = 3, n_filters: int = 256, n_mels: int = 80):
		super().__init__()
		self.n_layers = n_layers
		self.kernel_size = kernel_size
		self.n_filters = n_filters
		self.convs = nn.ModuleList([
			nn.Conv1d((n_filters if i > 0 else n_mels), n_filters, kernel_size, stride=(1 if i < n_layers-1 else 2), padding=1)
			for i in range(n_layers)
		])

	def forward(self, x: Tensor):
		'''
		Inputs:
			`x`: Tensor<Float>[B, T, M] input tensor (M is the mel-bins dim).
		
		Outputs:
			Tensor<Float>[B, T//2, E] output tensor, (E is the embedding dim and E = M // 2).
		'''
		x = x.transpose(-2, -1) # shift channels (M) to the front
		for i in range(self.n_layers):
			x = self.convs[i](x)
			x = nn.functional.gelu(x)
		return x.transpose(-2, -1) # shift channels (E) to the end