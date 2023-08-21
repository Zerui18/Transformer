import torch
from torch import Tensor

def softmax1(x: Tensor) -> Tensor:
	exp = torch.exp(x)
	# add 1.0 to normalizing constant to allow for "silence"
	normalizing = torch.sum(x, -1, keepdim=True) + 1.0
	# scores is no longer a valid probability distribution
	scores = exp / normalizing
	return scores