import torch
from torch import Tensor


def softmax1(x: Tensor) -> Tensor:
	''' Compute softmax with +1 in the normalizing constant, allowing attention to attend to nothing.

	Args:
		x: ``Tensor[(..., D), float32]``: unnormalized logits along the last dimension.
	Returns:
		``Tensor[(..., D), float32]``: normalized scores (not a valid probability distribution).
	'''
	# subtract max for numerical stability before exp
	x_max = x.max(dim=-1, keepdim=True).values
	exp = torch.exp(x - x_max)
	# add exp(-max) to normalizing constant to allow for "silence"
	# (this is the +1 term, adjusted for the max subtraction)
	normalizing = torch.sum(exp, -1, keepdim=True) + torch.exp(-x_max)
	scores = exp / normalizing
	return scores
