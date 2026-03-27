import os
from pathlib import Path
from multiprocessing import Process, Value, Array

from apscheduler.schedulers.background import BackgroundScheduler

from .experiment import Experiment, ExperimentState, ExperimentConfig


def _run_experiment_child_process(name: str, directory: str, state: Value,
								  err_buffer: Array, config: ExperimentConfig) -> None:
	''' Entry point for the child process of an experiment.

	Args:
		name: ``str``: experiment name.
		directory: ``str``: experiment output directory.
		state: ``Value``: shared experiment state.
		err_buffer: ``Value``: shared error message buffer.
		config: ``ExperimentConfig``: experiment configuration.
	'''
	import torch
	torch.set_float32_matmul_precision('high')
	import pytorch_lightning
	pytorch_lightning.seed_everything(42)
	experiment = Experiment(name, directory, state, err_buffer, config)
	experiment.run()


def _launch_experiment_process(experiment: Experiment) -> Process:
	''' Create, start, and return a Process for the given experiment.

	Args:
		experiment: ``Experiment``: the experiment to run.
	Returns:
		``Process``: the started subprocess.
	'''
	process = Process(
		target=_run_experiment_child_process,
		args=(experiment.name, str(experiment.directory),
			  experiment._state, experiment._err_buffer, experiment.config))
	process.start()
	return process


class ExperimentManager:
	''' Manages a queue of experiments, running them sequentially in subprocesses.

	Maintains four queues (queued, completed, stopped, failed) and one active slot.
	The next queued experiment starts automatically when the current one finishes.
	In single_process mode, experiments run in the main process for easier debugging.

	Attributes:
		master_directory: ``Path``: root directory where all experiment folders are created.
		single_process: ``bool``: whether to run experiments in the main process.
		queued_experiments: ``list[Experiment]``: experiments waiting to run.
		completed_experiments: ``list[Experiment]``: successfully finished experiments.
		stopped_experiments: ``list[Experiment]``: user-stopped experiments.
		failed_experiments: ``list[Experiment]``: experiments that errored.
	'''

	def __init__(self, master_directory: str | Path, single_process: bool = False):
		''' Initialize the experiment manager.

		Args:
			master_directory: ``str | Path``: root directory for experiment outputs.
			single_process: ``bool``: run in main process (for debugging).
		'''
		self.single_process = single_process
		self.master_directory = Path(master_directory)
		self.completed_experiments: list[Experiment] = []
		self.queued_experiments: list[Experiment] = []
		self.stopped_experiments: list[Experiment] = []
		self.failed_experiments: list[Experiment] = []
		self._current_experiment: Experiment | None = None
		self._current_process: Process | None = None

		if self.single_process:
			import torch
			torch.set_float32_matmul_precision('high')
		else:
			self._checker = BackgroundScheduler()
			self._checker.add_job(self._check_current_experiment, 'interval', seconds=1)
			self._checker.start()

	@property
	def current_experiment(self) -> Experiment | None:
		''' The currently running experiment, or None. '''
		return self._current_experiment

	@current_experiment.setter
	def current_experiment(self, experiment: Experiment | None) -> None:
		self._current_experiment = experiment

	def create_and_append_experiment(self, name: str, config: ExperimentConfig) -> None:
		''' Create an Experiment from a config and enqueue it.

		Args:
			name: ``str``: experiment name (also used as subdirectory name).
			config: ``ExperimentConfig``: the experiment configuration.
		'''
		state = Value('i', ExperimentState.QUEUING)
		err_buffer = Array('c', 1024)
		directory = self.master_directory / name
		directory.mkdir(parents=True, exist_ok=True)
		experiment = Experiment(name, directory, state, err_buffer, config)
		self.enqueue(experiment)

	def enqueue(self, experiment: Experiment) -> None:
		''' Add an experiment to the queue. Starts it immediately if nothing is running.

		Args:
			experiment: ``Experiment``: the experiment to enqueue.
		'''
		self.queued_experiments.append(experiment)
		if self.current_experiment is None:
			self._run_next_in_queue()

	def stop_current_experiment(self) -> None:
		''' Signal the current experiment to stop and move it to the stopped queue. '''
		self.current_experiment.state = ExperimentState.STOPPED
		stopped = self._remove_current_experiment()
		self.stopped_experiments.append(stopped)

	def move_in_queue(self, src_index: int, dst_index: int) -> None:
		''' Reorder the queue by moving an experiment from one position to another.

		Args:
			src_index: ``int``: current position.
			dst_index: ``int``: target position.
		'''
		self.queued_experiments.insert(dst_index, self.queued_experiments.pop(src_index))

	def enqueue_stopped(self, index: int) -> None:
		''' Re-enqueue a stopped experiment.

		Args:
			index: ``int``: position in the stopped queue.
		'''
		experiment = self.stopped_experiments.pop(index)
		experiment.state = ExperimentState.QUEUING
		self.enqueue(experiment)

	def enqueue_all_stopped(self) -> None:
		''' Re-enqueue all stopped experiments. '''
		for experiment in self.stopped_experiments:
			experiment.state = ExperimentState.QUEUING
			self.enqueue(experiment)
		self.stopped_experiments = []

	def stop_queued(self, index: int) -> None:
		''' Remove an experiment from the queue and mark it stopped.

		Args:
			index: ``int``: position in the queued list.
		'''
		experiment = self.queued_experiments.pop(index)
		experiment.state = ExperimentState.STOPPED
		self.stopped_experiments.append(experiment)

	def stop_all_queued(self) -> None:
		''' Stop all queued experiments. '''
		for experiment in self.queued_experiments:
			experiment.state = ExperimentState.STOPPED
			self.stopped_experiments.append(experiment)
		self.queued_experiments = []

	def remove_stopped(self, index: int) -> None:
		''' Remove a stopped experiment and delete its folder.

		Args:
			index: ``int``: position in the stopped queue.
		'''
		self.stopped_experiments.pop(index).remove_exp_folder()

	def remove_all_stopped(self) -> None:
		''' Remove all stopped experiments and delete their folders. '''
		for experiment in self.stopped_experiments:
			experiment.remove_exp_folder()
		self.stopped_experiments = []

	def remove_failed(self, index: int) -> None:
		''' Remove a failed experiment and delete its folder.

		Args:
			index: ``int``: position in the failed queue.
		'''
		self.failed_experiments.pop(index).remove_exp_folder()

	def remove_all_failed(self) -> None:
		''' Remove all failed experiments and delete their folders. '''
		for experiment in self.failed_experiments:
			experiment.remove_exp_folder()
		self.failed_experiments = []

	# --- Internal ---

	def _run_next_in_queue(self) -> None:
		''' Start the next experiment in the queue if one exists. '''
		if len(self.queued_experiments) > 0:
			self._set_current_experiment(0)

	def _set_current_experiment(self, index: int) -> None:
		''' Pop experiment from queue at index, make it current, and run it. '''
		assert self.current_experiment is None and self._current_process is None, \
			'Cannot start an experiment while another is running.'
		self.current_experiment = self.queued_experiments.pop(index)
		if self.single_process:
			# run experiments sequentially in a loop (avoids recursive setter)
			while self.current_experiment is not None:
				self.current_experiment.run()
				if self.current_experiment.state == ExperimentState.COMPLETED:
					self.completed_experiments.append(self.current_experiment)
				elif self.current_experiment.state == ExperimentState.STOPPED:
					self.stopped_experiments.append(self.current_experiment)
				elif self.current_experiment.state == ExperimentState.FAILED:
					self.failed_experiments.append(self.current_experiment)
				# advance to next in queue or stop
				if self.queued_experiments:
					self.current_experiment = self.queued_experiments.pop(0)
				else:
					self.current_experiment = None
		else:
			self._current_process = _launch_experiment_process(self.current_experiment)

	def _remove_current_experiment(self) -> Experiment:
		''' Remove and return the current experiment, cleaning up its process. '''
		if self.single_process:
			exp = self.current_experiment
			self.current_experiment = None
			return exp
		assert self.current_experiment is not None and self._current_process is not None
		exp = self.current_experiment
		if self._current_process.is_alive():
			self._current_process.join(timeout=5.0)
		if self._current_process.is_alive():
			self._current_process.terminate()
			self._current_process.join(timeout=3.0)
		self._current_process = None
		self.current_experiment = None
		return exp

	def _check_current_experiment(self) -> None:
		''' Poll the current experiment state and handle completion/failure. '''
		if self.current_experiment is not None:
			if self.current_experiment.state == ExperimentState.FAILED:
				print(f'Experiment {self.current_experiment.name} failed.')
				self.failed_experiments.append(self._remove_current_experiment())
			elif self.current_experiment.state == ExperimentState.COMPLETED:
				print(f'Experiment {self.current_experiment.name} completed.')
				self.completed_experiments.append(self._remove_current_experiment())
