import abc
import random

import torch
from torch import Tensor
import torchmetrics


class BaseMetric(torchmetrics.Metric, abc.ABC):
	''' Abstract base for all zlab metrics.

	Wraps torchmetrics.Metric with subsampling support, a required-keys
	contract, and step/epoch frequency control.

	Attributes:
		requires: ``set[str]``: The keys this metric expects from model.produce().
		name: ``str``: Display name used as the log key (defaults to class name).
		frequency: ``str``: 'step' for per-batch updates, 'epoch' for end-of-epoch updates.
		num_samples: ``int``: For epoch metrics, how many samples to evaluate on.
		subsample_rate: ``float | None``: Fraction of updates to keep (None = keep all).
		is_differentiable: ``bool``: Whether this metric supports gradient flow.
		higher_is_better: ``bool``: Whether higher values indicate better performance.
		full_state_update: ``bool``: Always False; updates must be decomposable.
	'''

	is_differentiable: bool = False
	higher_is_better: bool = True
	full_state_update: bool = False

	@property
	@abc.abstractmethod
	def requires(self) -> set[str]:
		''' Return the set of keys this metric needs from model.produce(). '''
		...

	def __init__(self,
				 name: str | None = None,
				 frequency: str = 'step',
				 num_samples: int = 32,
				 subsample_rate: float | None = None):
		''' Initialize the base metric.

		Args:
			name: ``str | None``: display name for logging. Defaults to the class name.
			frequency: ``str``: 'step' (update per batch) or 'epoch' (update once at epoch end).
			num_samples: ``int``: for epoch metrics, number of samples to evaluate on.
			subsample_rate: ``float | None``: [0 < x <= 1 or None] rate at which to keep updates. None means all updates are used.
		'''
		super().__init__()
		if frequency not in ('step', 'epoch'):
			raise ValueError(f'frequency must be "step" or "epoch", got {frequency!r}')
		if subsample_rate is not None and not (0.0 < subsample_rate <= 1.0):
			raise ValueError(f'subsample_rate must be in (0, 1] or None, got {subsample_rate}')
		self.name = name or self.__class__.__name__
		self.frequency = frequency
		self.num_samples = num_samples
		self.subsample_rate = subsample_rate

	def update(self, **kwargs) -> None:
		''' Public update entry point; handles subsampling then delegates to _update().

		Args:
			**kwargs: ``dict[str, Any]``: Keyword arguments matching self.requires keys.
		'''
		if self.subsample_rate is not None and random.random() > self.subsample_rate:
			return
		self._update(**kwargs)

	@abc.abstractmethod
	def _update(self, **kwargs) -> None:
		''' Core update logic. Subclasses implement this to accumulate state.

		Args:
			**kwargs: ``dict[str, Any]``: Keyword arguments matching self.requires keys.
		'''
		...

	@abc.abstractmethod
	def compute(self) -> Tensor:
		''' Compute the metric value from accumulated state.

		Returns:
			``Tensor[(), float32]``: scalar metric value.
		'''
		...
