import torch
from torch import Tensor


def softmax1(x: Tensor) -> Tensor:
	''' Compute softmax with +1 in the normalizing constant, allowing attention to attend to nothing.

	Args:
		1. x: Tensor  [float32, (..., D)] unnormalized logits along the last dimension.
	Returns:
		scores: Tensor  [float32, (..., D)] normalized scores (not a valid probability distribution).
	'''
	exp = torch.exp(x)
	# add 1.0 to normalizing constant to allow for "silence"
	normalizing = torch.sum(exp, -1, keepdim=True) + 1.0
	scores = exp / normalizing
	return scores
