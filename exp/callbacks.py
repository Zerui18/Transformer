from __future__ import annotations

from typing import TYPE_CHECKING

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback

from .experiment import ExperimentState

if TYPE_CHECKING:
	from multiprocessing.sharedctypes import Synchronized, SynchronizedArray

# NOTE: this module imports pytorch_lightning at module level and must therefore
# only be imported inside the training (child) process — see Experiment._init_trainer.

class ExperimentStopper(Callback):
	''' A callback for stopping the trainer when the experiment is stopped.

	Properties:
		1. state: Synchronized  [int] the ExperimentState of the experiment, stored in a multiprocessing.Value shared with the parent process
	'''

	def __init__(self, state: Synchronized):
		self.state = state

	def check_should_stop(self) -> bool:
		''' Returns whether the parent process has requested a stop.

		Returns: should_stop: bool  True if the shared state was set to STOPPED
		'''
		if self.state is None:
			return False
		return self.state.value == ExperimentState.STOPPED

	def on_train_batch_end(self, trainer: Trainer, *args):
		if self.check_should_stop():
			trainer.should_stop = True

	def on_validation_batch_end(self, trainer: Trainer, *args):
		if self.check_should_stop():
			trainer.should_stop = True

class ProgressReporter(Callback):
	''' A callback that publishes live training progress to the parent process for the web UI.

	Properties:
		1. progress: SynchronizedArray | None  [int32, (4,)] shared [epoch, batch, batches_per_epoch, global_step]; None disables reporting
	'''

	def __init__(self, progress: SynchronizedArray | None):
		self.progress = progress

	def on_train_batch_end(self, trainer: Trainer, pl_module, outputs, batch, batch_idx: int):
		if self.progress is None:
			return
		# num_training_batches is inf for iterable datasets without a length
		n_batches = trainer.num_training_batches
		n_batches = 0 if n_batches == float('inf') else int(n_batches)
		# single slice assignment -> atomic under the array's lock (no torn snapshots in the parent)
		self.progress[:] = [int(trainer.current_epoch), int(batch_idx) + 1, n_batches, int(trainer.global_step)]
