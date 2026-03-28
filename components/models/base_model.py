import abc
from typing import Any

import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities import grad_norm
from torchmetrics import MetricCollection

from components.metrics.base_metric import BaseMetric


class BaseModel(pl.LightningModule, abc.ABC):
	''' Abstract base for all zlab models.

	Extends LightningModule with a produce()/supports() contract that separates
	forward computation from the training loop.

	Two kinds of metrics:
	- Step metrics (``metrics``): updated per-batch in train/val/test steps from step outputs.
	  Stored as MetricCollection module attributes so Lightning auto-handles device placement.
	- Epoch metrics (``epoch_metrics``): independently sample from the val dataloader
	  and call produce() at the end of each training epoch. Stored as a MetricCollection.

	Attributes:
		optimizer_hparams: ``dict[str, Any]``: Optimizer config with 'cls' key and kwargs.

	Subclasses must implement:
		- produce(batch, requested) -> dict[str, Any]
		- supports() -> set[str]
	'''

	def __init__(self,
				 optimizer: dict[str, Any] = {},
				 metrics: dict[str, list[BaseMetric]] | None = None,
				 epoch_metrics: list[BaseMetric] | None = None):
		''' Initialize the base model.

		Args:
			optimizer: ``dict[str, Any]``: Optimizer config; 'cls' names a torch.optim class. Defaults to AdamW with lr=5e-4.
			metrics: ``dict[str, list[BaseMetric]] | None``: Stage-keyed step metrics ('train', 'val', 'test').
			epoch_metrics: ``list[BaseMetric] | None``: Epoch-level metrics that sample from val dataloader independently.
		'''
		super().__init__()
		self.optimizer_hparams = {
			'cls': 'AdamW',
			'lr': 5e-4,
			**optimizer,
		}
		# build MetricCollections as proper nn.Module attributes for auto device placement
		metrics = metrics or {}
		for stage, metric_list in metrics.items():
			collection = MetricCollection(
				{m.name: m for m in metric_list},
				prefix=f'{stage}_')
			setattr(self, f'{stage}_metrics', collection)
		self._metric_stages = list(metrics.keys())

		epoch_metrics = epoch_metrics or []
		self.epoch_metric_collection = MetricCollection(
			{m.name: m for m in epoch_metrics},
			prefix='epoch_')
		self._epoch_metric_list = epoch_metrics  # keep reference for num_samples access

	def _get_step_metrics(self, stage: str) -> MetricCollection | None:
		''' Get the MetricCollection for a stage, or None if not configured. '''
		return getattr(self, f'{stage}_metrics', None)

	def _validate_metrics(self) -> None:
		''' Check that every metric's requires is a subset of supports(). '''
		supported = self.supports()
		for stage in self._metric_stages:
			collection = self._get_step_metrics(stage)
			if collection is None:
				continue
			for name, metric in collection.items():
				missing = metric.requires - supported
				if missing:
					raise ValueError(
						f'{name} (stage={stage}) requires keys '
						f'{missing} but model only supports {supported}')
		for metric in self._epoch_metric_list:
			missing = metric.requires - supported
			if missing:
				raise ValueError(
					f'{metric.name} (epoch) requires keys '
					f'{missing} but model only supports {supported}')

	@abc.abstractmethod
	def produce(self, batch: dict[str, Any], requested: set[str]) -> dict[str, Any]:
		''' Produce requested outputs from a batch.

		Args:
			batch: ``dict[str, Any]``: Input batch from the dataloader.
			requested: ``set[str]``: Which outputs to compute.

		Returns:
			``dict[str, Any]``: At minimum {'loss': Tensor} when 'loss' is requested.

		May support expensive keys (e.g. 'decoded_greedy') for epoch metrics.
		'''
		...

	@abc.abstractmethod
	def supports(self) -> set[str]:
		''' Return the set of all output keys this model can produce.

		Returns:
			``set[str]``: e.g. {'loss', 'y_pred', 'y_true', 'decoded_greedy'}.
		'''
		...

	# --- Training loop ---

	def _gather_requested(self, stage: str) -> set[str]:
		''' Collect 'loss' + all step metric requires for the given stage.

		Args:
			stage: ``str``: One of 'train', 'val', 'test'.

		Returns:
			``set[str]``: Keys to request from produce().
		'''
		requested = {'loss'}
		collection = self._get_step_metrics(stage)
		if collection is not None:
			for metric in collection.values():
				requested |= metric.requires
		return requested

	def _step(self, batch: dict[str, Any], stage: str) -> torch.Tensor:
		''' Shared logic for training/validation/test steps.

		Args:
			batch: ``dict[str, Any]``: Input batch from the dataloader.
			stage: ``str``: One of 'train', 'val', 'test'.

		Returns:
			``Tensor[(), float32]``: The scalar loss.
		'''
		requested = self._gather_requested(stage)
		outputs = self.produce(batch, requested)
		loss = outputs['loss']

		self.log(f'{stage}_loss', loss, prog_bar=True)

		# update step metrics via MetricCollection
		collection = self._get_step_metrics(stage)
		if collection is not None:
			# build kwargs for all metrics (union of requires)
			all_keys: set[str] = set()
			for metric in collection.values():
				all_keys |= metric.requires
			metric_kwargs = {k: outputs[k] for k in all_keys}
			self.log_dict(collection(**metric_kwargs), prog_bar=True, on_step=True, on_epoch=True)

		return loss

	def _run_epoch_metrics(self) -> None:
		''' Run epoch metrics by sampling from the val dataloader and calling produce(). '''
		if not self._epoch_metric_list or not hasattr(self, '_val_dataloader'):
			return

		# gather all epoch metric requires
		requested: set[str] = set()
		for metric in self._epoch_metric_list:
			requested |= metric.requires
		max_samples = max(m.num_samples for m in self._epoch_metric_list)
		samples_seen = 0

		self.eval()
		with torch.inference_mode():
			for batch in self._val_dataloader:
				if samples_seen >= max_samples:
					break
				batch = {k: v.to(self.device) for k, v in batch.items()}
				# slice batch down if it would exceed max_samples
				remaining = max_samples - samples_seen
				batch_size = next(iter(batch.values())).size(0)
				if batch_size > remaining:
					batch = {k: v[:remaining] for k, v in batch.items()}
					batch_size = remaining
				outputs = self.produce(batch, requested)
				for metric in self._epoch_metric_list:
					if samples_seen < metric.num_samples:
						metric_kwargs = {k: outputs[k] for k in metric.requires}
						metric.update(**metric_kwargs)
				samples_seen += batch_size

		# log and reset via MetricCollection
		self.log_dict(self.epoch_metric_collection.compute(), prog_bar=True)
		self.epoch_metric_collection.reset()

	def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
		''' Training step: produce outputs, compute loss, update step metrics. '''
		return self._step(batch, 'train')

	def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
		''' Validation step: produce outputs, compute loss, update step metrics. '''
		return self._step(batch, 'val')

	def test_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
		''' Test step: produce outputs, compute loss, update step metrics. '''
		return self._step(batch, 'test')

	def configure_optimizers(self):
		''' Configure optimizer from self.optimizer_hparams using the nested dict pattern. '''
		hparams = dict(self.optimizer_hparams)
		cls_name = hparams.pop('cls')
		opt_class = getattr(torch.optim, cls_name)
		return opt_class(self.parameters(), **hparams)

	# --- Epoch hooks ---

	def on_train_epoch_end(self) -> None:
		''' Run epoch metrics at the end of each training epoch. '''
		if self.global_step > 0:
			self._run_epoch_metrics()

	def on_before_optimizer_step(self, optimizer) -> None:
		''' Log gradient norms before optimizer step. '''
		norms = grad_norm(self, 2)
		self.log_dict(norms)
