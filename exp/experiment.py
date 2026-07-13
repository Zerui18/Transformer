from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
	# heavy / optional imports, only needed for type checking
	from multiprocessing.sharedctypes import Synchronized, SynchronizedArray
	from torch.utils.data import DataLoader
	from pytorch_lightning import LightningModule, Trainer

# NOTE: torch / pytorch_lightning are intentionally NOT imported at module level.
# The web server (exp/server.py) and manager (exp/manager.py) import this module
# in the parent process, which may run on a machine without torch installed.
# All heavy imports happen lazily inside the _init_* methods, which only run in
# the training (child) process.

def _expanded_node_count(root: Any, budget: int) -> int:
	''' Counts the nodes a structure expands to when serialized (shared YAML-alias references count once per occurrence, exactly as JSON serialization would expand them).

	Args:
		1. root: Any  the parsed YAML structure
		2. budget: int  [> 0] stop counting once this many nodes have been seen
	Returns: count: int  the node count, capped just above budget

	Terminates even on recursive (cyclic) alias structures because it aborts at the budget.
	'''
	count = 0
	stack = [root]
	while stack:
		node = stack.pop()
		count += 1
		if count > budget:
			return count
		if isinstance(node, dict):
			stack.extend(node.keys())
			stack.extend(node.values())
		elif isinstance(node, (list, tuple, set)):
			stack.extend(node)
	return count

class ExperimentState:
	''' An enum for the state of an experiment.

	Properties:
		1. QUEUING: int  waiting to be run
		2. RUNNING: int  currently running
		3. COMPLETED: int  completed successfully
		4. STOPPED: int  stopped by user
		5. FAILED: int  failed due to an error
	'''
	QUEUING = 0 # waiting to be run
	RUNNING = 1 # currently running
	COMPLETED = 2 # completed successfully
	STOPPED = 3 # stopped by user
	FAILED = 4 # failed due to an error

class ExperimentConfig:
	''' A class for storing the config for an experiment.

	Properties:
		1. dls_config: dict  the dataloaders config
		2. model_config: dict  the model config
		3. trainer_config: dict  the trainer config
		4. resume_from_directory: str | None  experiment directory to resume from, None to create a new one
		5. resume_from_checkpoint: str | None  checkpoint path to resume from, None to start fresh
	'''

	def __init__(self, dls_config: dict, model_config: dict, trainer_config: dict, resume_from_directory: str | None = None, resume_from_checkpoint: str | None = None):
		''' Creates an ExperimentConfig object.

		Args:
			1. dls_config: dict  the dataloaders config
			2. model_config: dict  the model config
			3. trainer_config: dict  the trainer config
			4. resume_from_directory: str | None  [None] the path to the experiment directory to resume from; if None, a new experiment directory will be created
			5. resume_from_checkpoint: str | None  [None] the name of the checkpoint to resume from; if None, no checkpoint will be loaded
		'''
		self.dls_config = dls_config
		self.model_config = model_config
		self.trainer_config = trainer_config
		self.resume_from_directory = resume_from_directory
		self.resume_from_checkpoint = resume_from_checkpoint

	@staticmethod
	def from_config_files(model_config_file: str, dls_config_file: str, trainer_config_file: str, resume_from_directory: str | None = None, resume_from_checkpoint: str | None = None) -> 'ExperimentConfig':
		''' Returns an ExperimentConfig object from the given config files.

		Args:
			1. model_config_file: str  the path to the model config file
			2. dls_config_file: str  the path to the dataloaders config file
			3. trainer_config_file: str  the path to the trainer config file
			4. resume_from_directory: str | None  [None] the path to the experiment directory to resume from; if None, a new experiment directory will be created
			5. resume_from_checkpoint: str | None  [None] the name of the checkpoint to resume from; if None, no checkpoint will be loaded
		Returns: config: ExperimentConfig  the loaded config
		'''
		with open(model_config_file, 'r') as f:
			model_config = yaml.safe_load(f)
		with open(dls_config_file, 'r') as f:
			dls_config = yaml.safe_load(f)
		with open(trainer_config_file, 'r') as f:
			trainer_config = yaml.safe_load(f)
		config = ExperimentConfig(dls_config, model_config, trainer_config, resume_from_directory, resume_from_checkpoint)
		return config

	@staticmethod
	def from_yaml_strings(model_yaml: str | bytes, dls_yaml: str | bytes, trainer_yaml: str | bytes, resume_from_directory: str | None = None, resume_from_checkpoint: str | None = None) -> 'ExperimentConfig':
		''' Returns an ExperimentConfig object parsed from raw YAML strings (as submitted through the web UI / REST API).

		Args:
			1. model_yaml: str | bytes  raw YAML text of the model config
			2. dls_yaml: str | bytes  raw YAML text of the dataloaders config
			3. trainer_yaml: str | bytes  raw YAML text of the trainer config
			4. resume_from_directory: str | None  [None] the path to the experiment directory to resume from
			5. resume_from_checkpoint: str | None  [None] the name of the checkpoint to resume from
		Returns: config: ExperimentConfig  the parsed config

		Raises ValueError with a helpful message if any YAML does not parse to a mapping.
		'''
		configs = {}
		for key, raw in [('model', model_yaml), ('dls', dls_yaml), ('trainer', trainer_yaml)]:
			try:
				parsed = yaml.safe_load(raw)
			except yaml.YAMLError as e:
				raise ValueError(f'Invalid YAML in {key} config: {e}')
			if not isinstance(parsed, dict):
				raise ValueError(f'The {key} config must be a YAML mapping, got {type(parsed).__name__}.')
			# YAML aliases expand when the config is later JSON-serialized — reject
			# structures that would blow up (alias-expansion bombs)
			if _expanded_node_count(parsed, 100_000) > 100_000:
				raise ValueError(f'The {key} config expands to an unreasonable size (possible YAML alias bomb).')
			configs[key] = parsed
		return ExperimentConfig(configs['dls'], configs['model'], configs['trainer'], resume_from_directory, resume_from_checkpoint)

	@staticmethod
	def resuming_from_directory(directory: str, checkpoint_name: str | None = None) -> 'ExperimentConfig':
		''' Returns an ExperimentConfig object for resuming an experiment from a directory.

		Args:
			1. directory: str  the directory to resume from
			2. checkpoint_name: str | None  [None] the name of the checkpoint to resume from; if None, no checkpoint will be loaded
		Returns: config: ExperimentConfig  the loaded config
		'''
		directory = Path(directory)
		model_config_file = directory / 'model.yaml'
		dls_config_file = directory / 'dls.yaml'
		trainer_config_file = directory / 'trainer.yaml'
		if checkpoint_name is None:
			resume_from_checkpoint = None
		else:
			resume_from_checkpoint = directory / 'checkpoints' / checkpoint_name
		return ExperimentConfig.from_config_files(model_config_file, dls_config_file, trainer_config_file, directory, resume_from_checkpoint)


class Experiment:
	''' A class for running an experiment.

	Properties:
		1. name: str  the name of the experiment
		2. directory: Path  the experiment folder (configs, checkpoints, tensorboard events)
		3. config: ExperimentConfig  the experiment config
		4. state: int  the ExperimentState, backed by a multiprocessing.Value shared with the child process
		5. err_buffer: str  the last error message, backed by a multiprocessing.Array shared with the child process
		6. progress: dict  live training progress {epoch, batch, batches_per_epoch, global_step}, backed by a shared multiprocessing.Array
		7. launched_at: float | None  unix time when the manager launched this experiment (parent-side only)
		8. finished_at: float | None  unix time when the experiment left the running slot (parent-side only)
	'''

	### JIT INIT RESOURCES ###
	dls: dict[str, DataLoader] = None
	model: LightningModule = None
	trainer: Trainer = None

	### SHARED ###
	_state: Synchronized
	_err_buffer: SynchronizedArray
	_progress: SynchronizedArray | None

	### PARENT-SIDE BOOKKEEPING ###
	launched_at: float | None = None
	finished_at: float | None = None

	@property
	def state(self) -> int:
		return self._state.value
	@state.setter
	def state(self, value: int):
		self._state.value = value

	@property
	def err_buffer(self) -> str:
		return self._err_buffer.value.decode('utf-8', errors='replace')
	@err_buffer.setter
	def err_buffer(self, value: str):
		raw = value.encode('utf-8')[:len(self._err_buffer) - 1]
		# clear the whole buffer first so a shorter message doesn't leave stale bytes behind
		self._err_buffer[:] = b'\x00' * len(self._err_buffer)
		self._err_buffer[:len(raw)] = raw

	@property
	def progress(self) -> dict[str, int]:
		if self._progress is None:
			return {'epoch': 0, 'batch': 0, 'batches_per_epoch': 0, 'global_step': 0}
		epoch, batch, batches_per_epoch, global_step = self._progress[:]
		return {'epoch': epoch, 'batch': batch, 'batches_per_epoch': batches_per_epoch, 'global_step': global_step}

	def __init__(self, name: str, directory: str, state: Synchronized, err_buffer: SynchronizedArray, config: ExperimentConfig, progress: SynchronizedArray | None = None):
		''' Creates an Experiment object.

		Args:
			1. name: str  the name of the experiment
			2. directory: str  the experiment folder path
			3. state: Synchronized  [int] the state of the experiment, stored in a multiprocessing.Value('i')
			4. err_buffer: SynchronizedArray  [bytes] the error buffer, stored in a multiprocessing.Array('c')
			5. config: ExperimentConfig  the config for the experiment
			6. progress: SynchronizedArray | None  [None] [int32, (4,)] shared [epoch, batch, batches_per_epoch, global_step] live progress
		'''
		self._state = state
		self._err_buffer = err_buffer
		self._progress = progress
		self.name = name
		self.directory = Path(directory)
		self.config = config

	def run(self):
		''' Run the experiment.

		Note: This performs the heavy imports (torch, pytorch_lightning) and should only be called in the training (child) process, or in single-process mode.
		'''
		self.err_buffer = '' # clear error buffer
		try:
			self._init_resources()
			# compare-and-set under the shared value's lock: a stop requested during
			# initialization must not be overwritten by RUNNING
			with self._state.get_lock():
				if self._state.value == ExperimentState.STOPPED:
					return
				self._state.value = ExperimentState.RUNNING
			self.model._val_dataloader = self.dls['valid']
			self.trainer.fit(self.model, self.dls['train'], self.dls['valid'], ckpt_path=self.config.resume_from_checkpoint)
			# only set completed if not stopped (again compare-and-set: the parent may
			# have written STOPPED between fit() returning and this line)
			with self._state.get_lock():
				if self._state.value == ExperimentState.RUNNING:
					self._state.value = ExperimentState.COMPLETED
		except Exception as e:
			# capture any runtime exception & save to error buffer
			self.state = ExperimentState.FAILED
			self.err_buffer = f'{type(e).__name__}: {e}'
			raise e

	def _init_resources(self):
		''' Initialize all resources needed for the experiment.

		Note: This should only be called in the subprocess.
		'''
		# ensure experiment folder exists
		if self.config.resume_from_directory is None:
			self._init_exp_folder()
		else:
			self.directory = Path(self.config.resume_from_directory)
		# continue initializing resources
		self._init_dls()
		self._init_model()
		self._init_trainer()

	def _init_exp_folder(self):
		''' Initialize the experiment folder. '''
		self.directory.mkdir(parents=True, exist_ok=True)
		# save config files
		with open(self.directory / 'model.yaml', 'w') as f:
			yaml.safe_dump(self.config.model_config, f)
		with open(self.directory / 'dls.yaml', 'w') as f:
			yaml.safe_dump(self.config.dls_config, f)
		with open(self.directory / 'trainer.yaml', 'w') as f:
			yaml.safe_dump(self.config.trainer_config, f)
		# also create checkpoints folder
		(self.directory / 'checkpoints').mkdir(parents=True, exist_ok=True)

	def _init_dls(self):
		from torch.utils.data import DataLoader
		dls_config = self.config.dls_config
		dls = {}
		for name in ['train', 'valid']:
			config = dls_config[name]
			class_name = config['ds_class']
			init_args = config['ds_init_args']
			import datasets
			config_cls = getattr(datasets, class_name + 'Config')
			cls = getattr(datasets, class_name)
			ds: datasets.BaseDataset = cls(config_cls(**init_args)) # generic reference to dataset
			dls[name] = DataLoader(ds, collate_fn=ds.get_collate_function(), num_workers=8, pin_memory=True, drop_last=True, **config['dl_init_args'])
		self.dls = dls

	def _init_model(self):
		model_config = self.config.model_config
		# init tokenizer
		import toknizers
		tokenizer_config = model_config['tokenizer']
		tokenizer_class_name = tokenizer_config['class']
		tokenizer_init_args = tokenizer_config['init_args']
		tokenizer_cls = getattr(toknizers, tokenizer_class_name)
		tokenizer = tokenizer_cls(**tokenizer_init_args)
		# init model
		class_name = model_config['class']
		init_args = model_config['init_args']
		import models
		config_cls = getattr(models, class_name + 'Config')
		cls = getattr(models, class_name)
		self.model = cls(config_cls(**init_args), tokenizer=tokenizer)

	def _init_trainer(self):
		from pytorch_lightning import Trainer
		from pytorch_lightning.callbacks import ModelCheckpoint
		from pytorch_lightning.loggers import TensorBoardLogger
		from .callbacks import ExperimentStopper, ProgressReporter
		trainer_config = self.config.trainer_config
		# init callbacks
		self._experiment_stopper = ExperimentStopper(self._state)
		progress_reporter = ProgressReporter(self._progress)
		val_loss_ckpt = ModelCheckpoint(
			self.directory / 'checkpoints/',
			filename='model-{epoch}-{step}-{val_loss:.2f}',
			mode='min',
			monitor='val_loss',
			every_n_epochs=1,
			save_top_k=2,
			save_last=True)
		greedy_bleu_ckpt = ModelCheckpoint(
			self.directory / 'checkpoints/',
			filename='model-{epoch}-{step}-{bleu_greedy:.2f}',
			mode='max',
			monitor='bleu_greedy',
			every_n_epochs=1,
			save_top_k=2,
			save_last=True,
			save_on_train_epoch_end=True)
		bs_bleu_ckpt = ModelCheckpoint(
			self.directory / 'checkpoints/',
			filename='model-{epoch}-{step}-{bleu_bs16:.2f}',
			mode='max',
			monitor='bleu_bs16',
			every_n_epochs=1,
			save_top_k=2,
			save_last=True,
			save_on_train_epoch_end=True)
		# init logger
		logger = TensorBoardLogger(self.directory, name='', default_hp_metric=False, log_graph=False)
		self.trainer = Trainer(accelerator='gpu', devices=1,
								callbacks=[self._experiment_stopper, progress_reporter, val_loss_ckpt, greedy_bleu_ckpt, bs_bleu_ckpt],
								logger=logger,
								**trainer_config)

	def remove_exp_folder(self):
		''' Removes the experiment folder. '''
		shutil.rmtree(self.directory, ignore_errors=True)

	def get_dict_representation(self) -> dict[str, Any]:
		''' Returns a JSON-serializable dictionary representation of the experiment (as served by the REST API).

		Returns: representation: dict  keys: name, state, err_buffer, directory, progress, launched_at, finished_at, config{model, dl, trainer}
		'''
		return {
			'name': self.name,
			'state': self.state,
			'err_buffer': self.err_buffer,
			'directory': str(self.directory),
			'progress': self.progress,
			'launched_at': self.launched_at,
			'finished_at': self.finished_at,
			'config': {
				'model': self.config.model_config,
				'dls': self.config.dls_config,
				'trainer': self.config.trainer_config
			}
		}

	def __str__(self):
		return json.dumps(self.get_dict_representation(), indent=4)
