import yaml
import shutil
from pathlib import Path
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from multiprocessing import Value, Array
import json

class ExperimentState:
	''' An enum for the state of an experiment. '''
	QUEUING = 0 # waiting to be run
	RUNNING = 1 # currently running
	COMPLETED = 2 # completed successfully
	STOPPED = 3 # stopped by user
	FAILED = 4 # failed due to an error

class ExperimentStopper(Callback):
	''' A callback for stopping the trainer when the experiment is stopped. '''

	state: Value
	''' The state of the experiment, stored in a multiprocessing.Value object. '''

	def __init__(self, state: Value):
		self.state = state

	def check_should_stop(self):
		if self.state is None:
			return False
		return self.state.value == ExperimentState.STOPPED

	def on_train_batch_end(self, trainer: Trainer, *args):
		if self.check_should_stop():
			trainer.should_stop = True

	def on_validation_batch_end(self, trainer: Trainer, *args):
		if self.check_should_stop():
			trainer.should_stop = True

class ExperimentConfig:
	''' A class for storing the config for an experiment. '''

	def __init__(self, dls_config: dict, model_config: dict, trainer_config: dict, resume_from_directory: str = None, resume_from_checkpoint: str = None):
		''' Creates an ExperimentConfig object.

		Args:
			`dls_config (dict)`: The dataloaders config.
			`model_config (dict)`: The model config.
			`trainer_config (dict)`: The trainer config.
			`resume_from_directory (str, optional)`: The path to the experiment directory to resume from. If None, a new experiment directory will be created. Defaults to None.
			`resume_from_checkpoint (str, optional)`: The name of the checkpoint to resume from. If None, no checkpoint will be loaded. Defaults to None.
		'''
		self.dls_config = dls_config
		self.model_config = model_config
		self.trainer_config = trainer_config
		self.resume_from_directory = resume_from_directory
		self.resume_from_checkpoint = resume_from_checkpoint

	@staticmethod
	def from_config_files(model_config_file: str, dls_config_file: str, trainer_config_file: str, resume_from_directory: str = None, resume_from_checkpoint: str = None) -> 'ExperimentConfig':
		''' Returns an ExperimentConfig object from the given config files.
		
		Args:
			`model_config_file (str)`: The path to the model config file.
			`dls_config_file (str)`: The path to the dataloaders config file.
			`trainer_config_file (str)`: The path to the trainer config file.
			`resume_from_directory (str, optional)`: The path to the experiment directory to resume from. If None, a new experiment directory will be created. Defaults to None.
			`resume_from_checkpoint (str, optional)`: The name of the checkpoint to resume from. If None, no checkpoint will be loaded. Defaults to None.
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
	def resuming_from_directory(directory: str, checkpoint_name: str = None) -> 'ExperimentConfig':
		''' Returns an ExperimentConfig object for resuming an experiment from a directory.
		
		Args:
			`directory (str)`: The directory to resume from.
			`checkpoint_name (str, optional)`: The name of the checkpoint to resume from. If None, no checkpoint will be loaded. Defaults to None.
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
	''' A class for running an experiment. '''

	### JIT INIT RESOURCES ###
	dls: dict[str, DataLoader] = None
	model: LightningModule = None
	trainer: Trainer = None

	### SHARED ###
	_state: Value
	_err_buffer: Array

	@property
	def state(self) -> int:
		return self._state.value
	@state.setter
	def state(self, value: int):
		self._state.value = value

	@property
	def err_buffer(self) -> str:
		return self._err_buffer.value.decode('utf-8')
	@err_buffer.setter
	def err_buffer(self, value: str):
		bytes = value.encode('utf-8')
		min_len = min(len(bytes), len(self._err_buffer))
		self._err_buffer[:min_len] = bytes[:min_len]

	### PRIVATE ###
	_experiment_stopper: ExperimentStopper

	def __init__(self, name: str, directory: str, state: Value, err_buffer: Value, config: ExperimentConfig):
		''' Creates an Experiment object.

		Args:
			`name (str)`: The name of the experiment.
			`state (Value)`: The state of the experiment, stored in a multiprocessing.Value object.
			`config (ExperimentConfig)`: The config for the experiment.
		'''
		self._state = state
		self._err_buffer = err_buffer
		self.name = name
		self.directory = Path(directory)
		self.config = config

	def run(self):
		''' Run the experiment. '''
		self.err_buffer = '' # clear error buffer
		try:
			self._init_resources()
			self.state = ExperimentState.RUNNING
			self.trainer.fit(self.model, self.dls['train'], self.dls['valid'], ckpt_path=self.config.resume_from_checkpoint)
			# only set completed if not stopped
			if self.state == ExperimentState.RUNNING:
				self.state = ExperimentState.COMPLETED
		except Exception as e:
			# capture any runtime exception & save to error buffer
			self.state = ExperimentState.FAILED
			self.err_buffer = str(e)
			raise e

	def _init_resources(self):
		''' Initialize all resources needed for the experiment.

		Note: This should only be called in the subprocess.
		'''
		# ensure experiment folder exists
		if self.config.resume_from_directory is None:
			self._init_exp_folder()
		else:
			self.directory = self.config.resume_from_directory
		# if self._state is not None:
		#     # we're in child process
		#     # redirect stdout & stderr to log file in experiment folder
		#     import sys
		#     sys.stdout = open(self.directory / 'stdout.log', 'w')
		#     sys.stderr = open(self.directory / 'stderr.log', 'w')
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
		class_name = model_config['class']
		init_args = model_config['init_args']
		import models
		config_cls = getattr(models, class_name + 'Config')
		cls = getattr(models, class_name)
		self.model = cls(config_cls(**init_args))
		# compile model
		# TBD: compile on demand
		#import torch
		#self.model = torch.compile(self.model)

	def _init_trainer(self):
		trainer_config = self.config.trainer_config
		# init callbacks
		self._experiment_stopper = ExperimentStopper(self._state)
		val_loss_ckpt = ModelCheckpoint(
			self.directory / 'checkpoints/',
			filename='model-{epoch}-{step}-{val_loss:.2f}',
			mode='min',
			monitor='train_loss',
			every_n_train_steps=trainer_config.pop('model_checkpoint_interval'),
			save_top_k=2,
			save_last=True,)
		# init logger
		logger = TensorBoardLogger(self.directory, name='', default_hp_metric=False, log_graph=False)
		self.trainer = Trainer(accelerator='gpu', devices=1,
								callbacks=[self._experiment_stopper, val_loss_ckpt],
								logger=logger,
								**trainer_config)

	def remove_exp_folder(self):
		''' Removes the experiment folder. '''
		shutil.rmtree(self.directory)

	def get_dict_representation(self):
		''' Returns a dictionary representation of the experiment. '''
		return {
			'name': self.name,
			'state': self.state,
			'err_buffer': self.err_buffer,
			'config': {
				'model': self.config.model_config,
				'dl': self.config.dls_config,
				'trainer': self.config.trainer_config
			}
		}

	def __str__(self):
		return json.dumps(self.get_dict_representation(), indent=4)