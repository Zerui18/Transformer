import abc
from typing import Any

import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities import grad_norm

from components.metrics.base_metric import BaseMetric


class BaseModel(pl.LightningModule, abc.ABC):
	''' Abstract base for all zlab models.

	Extends LightningModule with a produce()/supports() contract that separates
	forward computation from the training loop. Metrics are stage-specific and
	support both per-step and per-epoch frequencies.

	Properties:
		1. optimizer_hparams: dict[str, Any]  optimizer config with 'cls' key and kwargs.
		2. metrics: dict[str, list[BaseMetric]]  stage name -> list of metrics for that stage.

	Subclasses must implement:
		- produce(batch, requested) -> dict[str, Any]
		- supports() -> set[str]
	'''

	def __init__(self,
				 optimizer: dict[str, Any] = {},
				 metrics: dict[str, list[BaseMetric]] | None = None):
		''' Initialize the base model.

		Args:
			1. optimizer: dict[str, Any]  optimizer config; 'cls' names a torch.optim class, remaining keys are passed as kwargs. Defaults to AdamW with lr=5e-4.
			2. metrics: dict[str, list[BaseMetric]] | None  stage-keyed metrics ('train', 'val', 'test'). None means no metrics.
		'''
		super().__init__()
		self.optimizer_hparams = {
			'cls': 'AdamW',
			'lr': 5e-4,
			**optimizer,
		}
		self.metrics = metrics or {}
		self._train_losses: list[float] = []
		self._val_losses: list[float] = []

	def _validate_metrics(self) -> None:
		''' Check that every metric's requires is a subset of supports(). '''
		supported = self.supports()
		for stage, metric_list in self.metrics.items():
			for metric in metric_list:
				missing = metric.requires - supported
				if missing:
					raise ValueError(
						f'{metric.name} (stage={stage}) requires keys '
						f'{missing} but model only supports {supported}')

	@abc.abstractmethod
	def produce(self, batch: dict[str, Any], requested: set[str]) -> dict[str, Any]:
		''' Produce requested outputs from a batch.

		Args:
			1. batch: dict[str, Any]  input batch from the dataloader.
			2. requested: set[str]  which outputs to compute.
		Returns:
			outputs: dict[str, Any]  at minimum {'loss': Tensor} when 'loss' is requested.

		Must always produce 'loss' when requested. Other keys are model-specific
		and consumed by metrics. Epoch metrics may request expensive keys (e.g.
		'decoded_greedy') that trigger autoregressive decoding.
		'''
		...

	@abc.abstractmethod
	def supports(self) -> set[str]:
		''' Return the set of all output keys this model can produce.

		Returns:
			keys: set[str]  e.g. {'loss', 'y_pred', 'y_true', 'decoded_greedy', 'decoded_beam'}.
		'''
		...

	# --- Metric helpers ---

	def _step_metrics(self, stage: str) -> list[BaseMetric]:
		''' Return only the step-frequency metrics for a stage. '''
		return [m for m in self.metrics.get(stage, []) if m.frequency == 'step']

	def _epoch_metrics(self, stage: str) -> list[BaseMetric]:
		''' Return only the epoch-frequency metrics for a stage. '''
		return [m for m in self.metrics.get(stage, []) if m.frequency == 'epoch']

	def _gather_requested(self, stage: str, frequency: str = 'step') -> set[str]:
		''' Collect the union of 'loss' and all metric requires for the given stage and frequency.

		Args:
			1. stage: str  one of 'train', 'val', 'test'.
			2. frequency: str  'step' or 'epoch'.
		Returns:
			requested: set[str]  keys to request from produce().
		'''
		requested = {'loss'}
		metrics = self._step_metrics(stage) if frequency == 'step' else self._epoch_metrics(stage)
		for metric in metrics:
			requested |= metric.requires
		return requested

	# --- Training loop (delegates to produce + metrics) ---

	def _step(self, batch: dict[str, Any], stage: str) -> torch.Tensor:
		''' Shared logic for training/validation/test steps. Only runs step-frequency metrics.

		Args:
			1. batch: dict[str, Any]  input batch from the dataloader.
			2. stage: str  one of 'train', 'val', 'test'.
		Returns:
			loss: Tensor  [float32, ()] the scalar loss.
		'''
		requested = self._gather_requested(stage, frequency='step')
		outputs = self.produce(batch, requested)
		loss = outputs['loss']

		self.log(f'{stage}_loss', loss, prog_bar=True)

		# update step-frequency metrics only
		for metric in self._step_metrics(stage):
			metric_kwargs = {k: outputs[k] for k in metric.requires}
			metric.update(**metric_kwargs)
			self.log(f'{stage}_{metric.name}', metric, prog_bar=True)

		return loss

	def _run_epoch_metrics(self, stage: str) -> None:
		''' Run epoch-frequency metrics by sampling from the val dataloader and calling produce.

		Args:
			1. stage: str  the stage to run epoch metrics for.
		'''
		epoch_metrics = self._epoch_metrics(stage)
		if not epoch_metrics or not hasattr(self, '_val_dataloader'):
			return

		requested = self._gather_requested(stage, frequency='epoch')
		max_samples = max(m.num_samples for m in epoch_metrics)
		samples_seen = 0

		self.eval()
		with torch.inference_mode():
			for batch in self._val_dataloader:
				if samples_seen >= max_samples:
					break
				# move batch to device
				batch = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in batch.items()}
				outputs = self.produce(batch, requested)
				for metric in epoch_metrics:
					if samples_seen < metric.num_samples:
						metric_kwargs = {k: outputs[k] for k in metric.requires}
						metric.update(**metric_kwargs)
				# count samples in this batch
				first_val = next(iter(batch.values()))
				samples_seen += first_val.size(0) if hasattr(first_val, 'size') else len(first_val)

		# log and reset
		for metric in epoch_metrics:
			value = metric.compute()
			self.log(f'{stage}_{metric.name}', value, prog_bar=True)
			metric.reset()

	def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
		''' Training step: produce outputs, compute loss, update step metrics.

		Args:
			1. batch: dict[str, Any]  input batch.
			2. batch_idx: int  index of this batch within the epoch.
		Returns:
			loss: Tensor  [float32, ()] scalar loss for backprop.
		'''
		loss = self._step(batch, 'train')
		self._train_losses.append(loss.item())
		return loss

	def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
		''' Validation step: produce outputs, compute loss, update step metrics.

		Args:
			1. batch: dict[str, Any]  input batch.
			2. batch_idx: int  index of this batch within the epoch.
		Returns:
			loss: Tensor  [float32, ()] scalar loss.
		'''
		loss = self._step(batch, 'val')
		self._val_losses.append(loss.item())
		return loss

	def test_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
		''' Test step: produce outputs, compute loss, update step metrics.

		Args:
			1. batch: dict[str, Any]  input batch.
			2. batch_idx: int  index of this batch within the epoch.
		Returns:
			loss: Tensor  [float32, ()] scalar loss.
		'''
		return self._step(batch, 'test')

	def configure_optimizers(self):
		''' Configure optimizer from self.optimizer_hparams using the nested dict pattern.

		The 'cls' key names a torch.optim class; all other keys are passed as kwargs.
		'''
		hparams = dict(self.optimizer_hparams)
		cls_name = hparams.pop('cls')
		opt_class = getattr(torch.optim, cls_name)
		return opt_class(self.parameters(), **hparams)

	# --- Epoch hooks ---

	def on_train_epoch_start(self) -> None:
		''' Reset per-epoch train loss accumulator. '''
		self._train_losses = []

	def on_validation_epoch_start(self) -> None:
		''' Reset per-epoch validation loss accumulator. '''
		self._val_losses = []

	def on_train_epoch_end(self) -> None:
		''' Log average training loss for the epoch. '''
		if self._train_losses:
			avg = sum(self._train_losses) / len(self._train_losses)
			print(f'Epoch {self.trainer.current_epoch} train loss: {avg:.4f}')

	def on_validation_epoch_end(self) -> None:
		''' Log average validation loss, then run epoch-frequency metrics. '''
		if self._val_losses:
			avg = sum(self._val_losses) / len(self._val_losses)
			print(f'Epoch {self.trainer.current_epoch} val loss: {avg:.4f}')
		# run epoch metrics (e.g. decoding BLEU) if any
		if self.global_step > 0:
			self._run_epoch_metrics('val')

	def on_before_optimizer_step(self, optimizer) -> None:
		''' Log gradient norms before optimizer step. '''
		norms = grad_norm(self, 2)
		self.log_dict(norms)
