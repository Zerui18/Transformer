import torch
from torch import Tensor

from components.metrics.base_metric import BaseMetric


class TokenAccuracyMetric(BaseMetric):
	''' Per-token classification accuracy accumulated over batches.

	Computes the fraction of correctly predicted tokens, averaged across
	the entire corpus.

	Properties:
		1. is_differentiable: bool  False — accuracy is non-differentiable.
		2. higher_is_better: bool  True — higher accuracy is better.
	'''

	is_differentiable: bool = False
	higher_is_better: bool = True

	@property
	def requires(self) -> set[str]:
		''' Keys needed from model.produce(): y_pred logits and y_true token ids. '''
		return {'y_pred', 'y_true'}

	def __init__(self, subsample_rate: float | None = None):
		''' Initialize the token accuracy metric.

		Args:
			1. subsample_rate: float | None  fraction of updates to keep. None = all.
		'''
		super().__init__(subsample_rate=subsample_rate)
		self.add_state('correct', default=torch.tensor(0, dtype=torch.long), dist_reduce_fx='sum')
		self.add_state('total', default=torch.tensor(0, dtype=torch.long), dist_reduce_fx='sum')

	def _update(self, y_pred: Tensor, y_true: Tensor) -> None:
		''' Accumulate correct/total token counts from a batch.

		Args:
			1. y_pred: Tensor  [float32, (B, T, V)] predicted logits.
			2. y_true: Tensor  [int64, (B, T)] ground truth token ids.
		'''
		B, T = y_true.shape
		pred_ids = y_pred.view(B * T, -1).argmax(dim=-1)  # (B*T,)
		true_ids = y_true.reshape(B * T)  # (B*T,)
		self.correct += (pred_ids == true_ids).sum()
		self.total += true_ids.numel()

	def compute(self) -> Tensor:
		''' Compute corpus-level token accuracy.

		Returns:
			accuracy: Tensor  [float32, ()] accuracy in [0, 1].
		'''
		if self.total == 0:
			return torch.tensor(0.0)
		return self.correct.float() / self.total.float()
