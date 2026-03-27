import torch
from torch import Tensor

from components.metrics.base_metric import BaseMetric


class TokenAccuracyMetric(BaseMetric):
	''' Per-token classification accuracy accumulated over batches.

	Computes the fraction of correctly predicted tokens, averaged across
	the entire corpus.

	Attributes:
		is_differentiable: ``bool``: False — accuracy is non-differentiable.
		higher_is_better: ``bool``: True — higher accuracy is better.
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
			subsample_rate: ``float | None``: fraction of updates to keep. None = all.
		'''
		super().__init__(subsample_rate=subsample_rate)
		self.add_state('correct', default=torch.tensor(0, dtype=torch.long), dist_reduce_fx='sum')
		self.add_state('total', default=torch.tensor(0, dtype=torch.long), dist_reduce_fx='sum')

	def _update(self, y_pred: Tensor, y_true: Tensor) -> None:
		''' Accumulate correct/total token counts from a batch.

		Args:
			y_pred: ``Tensor[(B, T, V), float32]``: predicted logits.
			y_true: ``Tensor[(B, T), int64]``: ground truth token ids.
		'''
		B, T = y_true.shape
		pred_ids = y_pred.view(B * T, -1).argmax(dim=-1)  # (B*T,)
		true_ids = y_true.reshape(B * T)  # (B*T,)
		self.correct += (pred_ids == true_ids).sum()
		self.total += true_ids.numel()

	def compute(self) -> Tensor:
		''' Compute corpus-level token accuracy.

		Returns:
			``Tensor[(), float32]``: accuracy in [0, 1].
		'''
		if self.total == 0:
			return torch.tensor(0.0)
		return self.correct.float() / self.total.float()
