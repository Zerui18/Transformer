import json
import shutil
import yaml
from pathlib import Path

from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from multiprocessing import Value, Array


class ExperimentState:
	''' Enum-like constants for experiment lifecycle states.

	Attributes:
		QUEUING: ``int``: waiting in queue to run.
		RUNNING: ``int``: currently executing.
		COMPLETED: ``int``: finished successfully.
		STOPPED: ``int``: stopped by the user.
		FAILED: ``int``: terminated due to an error.
	'''
	QUEUING = 0
	RUNNING = 1
	COMPLETED = 2
	STOPPED = 3
	FAILED = 4


class ExperimentStopper(Callback):
	''' Lightning callback that stops the trainer when the experiment state transitions to STOPPED.

	Attributes:
		state: ``Value``: shared multiprocessing.Value holding the ExperimentState int.
	'''

	def __init__(self, state: Value):
		''' Initialize the stopper callback.

		Args:
			state: ``Value``: shared experiment state.
		'''
		self.state = state

	def check_should_stop(self) -> bool:
		''' Return True if the experiment has been marked as STOPPED.

		Returns:
			``bool``: whether the trainer should stop.
		'''
		if self.state is None:
			return False
		return self.state.value == ExperimentState.STOPPED

	def on_train_batch_end(self, trainer: Trainer, *args) -> None:
		''' Check for stop signal after each training batch. '''
		if self.check_should_stop():
			trainer.should_stop = True

	def on_validation_batch_end(self, trainer: Trainer, *args) -> None:
		''' Check for stop signal after each validation batch. '''
		if self.check_should_stop():
			trainer.should_stop = True


class ExperimentConfig:
	''' Stores the three config dicts (model, dataset, training) for an experiment.

	Attributes:
		model_config: ``dict``: model class, hparams, tokenizer, and metrics configuration.
		dataset_config: ``dict``: dataset class, init args, and dataloader args per split.
		training_config: ``dict``: Lightning Trainer keyword arguments.
		resume_from_directory: ``str | None``: experiment directory to resume from.
		resume_from_checkpoint: ``str | None``: checkpoint path to resume from.
	'''

	def __init__(self, dataset_config: dict, model_config: dict, training_config: dict,
				 resume_from_directory: str | None = None,
				 resume_from_checkpoint: str | None = None):
		''' Create an ExperimentConfig.

		Args:
			dataset_config: ``dict``: dataset configuration per split.
			model_config: ``dict``: model class, hparams, tokenizer, metrics.
			training_config: ``dict``: Lightning Trainer kwargs.
			resume_from_directory: ``str | None``: path to resume experiment from.
			resume_from_checkpoint: ``str | None``: checkpoint name to resume from.
		'''
		self.dataset_config = dataset_config
		self.model_config = model_config
		self.training_config = training_config
		self.resume_from_directory = resume_from_directory
		self.resume_from_checkpoint = resume_from_checkpoint

	# backward-compat aliases for old field names
	@property
	def dls_config(self) -> dict:
		''' Backward-compatible alias for dataset_config. '''
		return self.dataset_config

	@property
	def trainer_config(self) -> dict:
		''' Backward-compatible alias for training_config. '''
		return self.training_config

	@staticmethod
	def from_config_files(model_config_file: str, dataset_config_file: str,
						  training_config_file: str,
						  resume_from_directory: str | None = None,
						  resume_from_checkpoint: str | None = None) -> 'ExperimentConfig':
		''' Load an ExperimentConfig from three YAML files.

		Args:
			model_config_file: ``str``: path to model.yaml.
			dataset_config_file: ``str``: path to dataset.yaml.
			training_config_file: ``str``: path to training.yaml.
			resume_from_directory: ``str | None``: experiment directory to resume from.
			resume_from_checkpoint: ``str | None``: checkpoint name to resume from.
		Returns:
			``ExperimentConfig``: the loaded config.
		'''
		with open(model_config_file, 'r') as f:
			model_config = yaml.safe_load(f)
		with open(dataset_config_file, 'r') as f:
			dataset_config = yaml.safe_load(f)
		with open(training_config_file, 'r') as f:
			training_config = yaml.safe_load(f)
		return ExperimentConfig(dataset_config, model_config, training_config,
								resume_from_directory, resume_from_checkpoint)

	@staticmethod
	def resuming_from_directory(directory: str,
								checkpoint_name: str | None = None) -> 'ExperimentConfig':
		''' Load an ExperimentConfig for resuming from a saved experiment directory.

		Args:
			directory: ``str``: path to the experiment directory.
			checkpoint_name: ``str | None``: name of the checkpoint file to resume from.
		Returns:
			``ExperimentConfig``: the loaded config.
		'''
		d = Path(directory)
		# try new filenames first, fall back to old
		model_file = d / 'model.yaml'
		dataset_file = d / 'dataset.yaml' if (d / 'dataset.yaml').exists() else d / 'dls.yaml'
		training_file = d / 'training.yaml' if (d / 'training.yaml').exists() else d / 'trainer.yaml'
		ckpt = None if checkpoint_name is None else str(d / 'checkpoints' / checkpoint_name)
		return ExperimentConfig.from_config_files(
			str(model_file), str(dataset_file), str(training_file), str(d), ckpt)


class Experiment:
	''' Manages the full lifecycle of a single experiment: resource init, training, and cleanup.

	Experiments are designed to run in a subprocess. Resources (dataloaders, model, trainer)
	are JIT-initialized in the subprocess via run(). State is shared with the parent
	process through multiprocessing.Value/Array.

	Attributes:
		name: ``str``: human-readable experiment name.
		directory: ``Path``: experiment output directory.
		config: ``ExperimentConfig``: the experiment configuration.
		dls: ``dict[str, DataLoader] | None``: JIT-initialized dataloaders.
		model: ``LightningModule | None``: JIT-initialized model.
		trainer: ``Trainer | None``: JIT-initialized Lightning Trainer.
	'''

	dls: dict[str, DataLoader] | None = None
	model: LightningModule | None = None
	trainer: Trainer | None = None

	_state: Value
	_err_buffer: Array

	@property
	def state(self) -> int:
		''' The current ExperimentState value (shared across processes). '''
		return self._state.value

	@state.setter
	def state(self, value: int) -> None:
		self._state.value = value

	@property
	def err_buffer(self) -> str:
		''' The error message buffer (shared across processes). '''
		return self._err_buffer.value.decode('utf-8')

	@err_buffer.setter
	def err_buffer(self, value: str) -> None:
		encoded = value.encode('utf-8')
		n = min(len(encoded), len(self._err_buffer))
		self._err_buffer[:n] = encoded[:n]

	def __init__(self, name: str, directory: str | Path, state: Value,
				 err_buffer: Array, config: ExperimentConfig):
		''' Create an Experiment.

		Args:
			name: ``str``: experiment name.
			directory: ``str | Path``: output directory.
			state: ``Value``: shared multiprocessing state.
			err_buffer: ``Array``: shared error message buffer.
			config: ``ExperimentConfig``: the experiment configuration.
		'''
		self._state = state
		self._err_buffer = err_buffer
		self.name = name
		self.directory = Path(directory)
		self.config = config

	def run(self) -> None:
		''' Execute the experiment: init resources, fit the model, update state on completion or failure. '''
		self.err_buffer = ''
		try:
			self._init_resources()
			self.state = ExperimentState.RUNNING
			# attach val dataloader for model-level BLEU reporting
			self.model._val_dataloader = self.dls['valid']
			self.trainer.fit(self.model, self.dls['train'], self.dls['valid'],
							 ckpt_path=self.config.resume_from_checkpoint)
			if self.state == ExperimentState.RUNNING:
				self.state = ExperimentState.COMPLETED
		except Exception as e:
			self.state = ExperimentState.FAILED
			self.err_buffer = str(e)
			raise

	def _init_resources(self) -> None:
		''' Initialize experiment folder, dataloaders, model, and trainer. '''
		if self.config.resume_from_directory is None:
			self._init_exp_folder()
		else:
			self.directory = Path(self.config.resume_from_directory)
		self._init_dls()
		self._init_model()
		self._init_trainer()

	def _init_exp_folder(self) -> None:
		''' Create the experiment directory and save config files. '''
		self.directory.mkdir(parents=True, exist_ok=True)
		with open(self.directory / 'model.yaml', 'w') as f:
			yaml.safe_dump(self.config.model_config, f)
		with open(self.directory / 'dataset.yaml', 'w') as f:
			yaml.safe_dump(self.config.dataset_config, f)
		with open(self.directory / 'training.yaml', 'w') as f:
			yaml.safe_dump(self.config.training_config, f)
		(self.directory / 'checkpoints').mkdir(parents=True, exist_ok=True)

	def _init_dls(self) -> None:
		''' Initialize dataloaders from the dataset config using the components registry. '''
		import components
		ds_config = self.config.dataset_config
		dls = {}
		for name in ['train', 'valid']:
			split = ds_config[name]
			class_name = split.get('cls') or split.get('ds_class')
			dl_args = split.get('dataloader', split.get('dl_init_args', {}))
			# dataset kwargs: flat keys excluding reserved ones
			reserved = {'cls', 'ds_class', 'ds_init_args', 'dl_init_args', 'dataloader'}
			ds_args = split.get('ds_init_args', {})
			ds_args.update({k: v for k, v in split.items() if k not in reserved})
			cls = components.datasets_registry[class_name]
			ds = cls(**ds_args)
			dls[name] = DataLoader(ds, collate_fn=ds.get_collate_function(),
								   num_workers=8, pin_memory=True, drop_last=True,
								   **dl_args)
		self.dls = dls

	def _init_model(self) -> None:
		''' Initialize the model, tokenizer, and metrics from the model config using registries. '''
		import components
		mc = self.config.model_config
		# tokenizer: flat dict with 'cls' key, remaining keys are constructor kwargs
		tokenizer = None
		if 'tokenizer' in mc:
			tc = mc['tokenizer']
			tok_cls_name = tc.get('cls')
			tok_args = {k: v for k, v in tc.items() if k != 'cls'}
			tokenizer = components.tokenizers_registry[tok_cls_name](**tok_args)
		# metrics: stage name -> list of metric instances
		metrics: dict[str, list] = {}
		if 'metrics' in mc:
			for stage, stage_metrics in mc['metrics'].items():
				metrics[stage] = []
				for mcfg in stage_metrics:
					m_cls_name = mcfg['cls']
					m_args = {k: v for k, v in mcfg.items() if k != 'cls'}
					# inject tokenizer into metric if configured as dict
					if 'tokenizer' in m_args and isinstance(m_args['tokenizer'], dict):
						t = m_args['tokenizer']
						t_cls = components.tokenizers_registry[t.get('cls') or t.get('class')]
						m_args['tokenizer'] = t_cls(**{k: v for k, v in t.items() if k not in ('cls', 'class')})
					metrics[stage].append(components.metrics_registry[m_cls_name](**m_args))
		# optimizer: nested dict with 'cls' key
		optimizer = dict(mc.get('optimizer', {}))
		# model kwargs: everything except reserved keys
		reserved = {'cls', 'tokenizer', 'metrics', 'optimizer', 'checkpoints'}
		init_args = {k: v for k, v in mc.items() if k not in reserved}
		# model
		model_cls_name = mc['cls']
		self.model = components.models_registry[model_cls_name](
			**init_args, tokenizer=tokenizer, optimizer=optimizer, metrics=metrics)

	def _init_trainer(self) -> None:
		''' Initialize the Lightning Trainer with callbacks, logger, and config-driven checkpoints. '''
		self._experiment_stopper = ExperimentStopper(self._state)
		# build checkpoint callbacks from model config (or default to val_loss)
		checkpoint_configs = self.config.model_config.get('checkpoints', [
			{'monitor': 'val_loss', 'mode': 'min'},
		])
		checkpoint_callbacks = []
		for ckpt_cfg in checkpoint_configs:
			monitor = ckpt_cfg['monitor']
			mode = ckpt_cfg.get('mode', 'min')
			save_top_k = ckpt_cfg.get('save_top_k', 2)
			filename = ckpt_cfg.get('filename', f'model-{{epoch}}-{{step}}-{{{monitor}:.2f}}')
			checkpoint_callbacks.append(ModelCheckpoint(
				self.directory / 'checkpoints/',
				filename=filename,
				mode=mode, monitor=monitor,
				every_n_epochs=1, save_top_k=save_top_k, save_last=True,
				save_on_train_epoch_end=False))
		logger = TensorBoardLogger(self.directory, name='', default_hp_metric=False, log_graph=False)
		self.trainer = Trainer(
			accelerator='gpu', devices=1,
			callbacks=[self._experiment_stopper, *checkpoint_callbacks],
			logger=logger,
			**self.config.training_config)

	def remove_exp_folder(self) -> None:
		''' Delete the experiment output directory. '''
		shutil.rmtree(self.directory)

	def get_dict_representation(self) -> dict:
		''' Return a JSON-serializable summary of this experiment.

		Returns:
			``dict``: keys: name, state, err_buffer, config.
		'''
		return {
			'name': self.name,
			'state': self.state,
			'err_buffer': self.err_buffer,
			'config': {
				'model': self.config.model_config,
				'dataset': self.config.dataset_config,
				'training': self.config.training_config,
			}
		}

	def __str__(self) -> str:
		return json.dumps(self.get_dict_representation(), indent=4)
