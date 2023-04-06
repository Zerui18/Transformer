import yaml
import shutil
from pathlib import Path
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from multiprocessing import Value
import json

class ExperimentState:
    ''' An enum for the state of an experiment. '''
    QUEUING = 0 # waiting to be run
    RUNNING = 1 # currently running
    STOPPED = 2 # stopped by user (will not be run until changed to queuing)
    COMPLETED = 3 # finished running

class ExperimentStopper(Callback):
    ''' A callback for stopping the trainer when the experiment is stopped. '''

    state: Value
    ''' The state of the experiment, stored in a multiprocessing.Value object. '''

    def __init__(self, state: Value):
        self.state = state

    def check_should_stop(self):
        if self.state is None:
            return False
        return self.state.value != ExperimentState.RUNNING

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
            `resume_from_checkpoint (str, optional)`: The path to the checkpoint to resume from. If None, no checkpoint will be loaded. Defaults to None.
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
            `resume_from_checkpoint (str, optional)`: The path to the checkpoint to resume from. If None, no checkpoint will be loaded. Defaults to None.
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
    
    @property
    def state(self) -> int:
        return self._state.value
    @state.setter
    def state(self, value: int):
        self._state.value = value

    ### PRIVATE ###
    _experiment_stopper: ExperimentStopper

    def __init__(self, name: str, state: Value, config: ExperimentConfig):
        ''' Creates an Experiment object.

        Args:
            `name (str)`: The name of the experiment.
            `state (Value)`: The state of the experiment, stored in a multiprocessing.Value object.
            `config (ExperimentConfig)`: The config for the experiment.
        '''
        self._state = state
        self.name = name
        self.config = config

    def start(self):
        self.state = ExperimentState.QUEUING
    
    def stop(self):
        self.state = ExperimentState.STOPPED
    
    def run(self):
        self.state = ExperimentState.RUNNING
        self.trainer.fit(self.model, self.dls['train'], self.dls['valid'], ckpt_path=self.config.resume_from_checkpoint)
        # only set completed if not stopped
        if self.state == ExperimentState.RUNNING:
            self.state = ExperimentState.COMPLETED

    def init_resources(self):
        ''' Initialize all resources needed for the experiment.

        Note: This should only be called in the subprocess.
        '''
        if self.config.resume_from_directory is None:
            self.init_exp_folder()
        else:
            self.exp_folder = self.config.resume_from_directory
        self.init_dls()
        self.init_model()
        self.init_trainer()

    def init_exp_folder(self):
        ''' Initialize the experiment folder. '''
        exp_folder = Path(f'experiments/{self.name}')
        exp_folder.mkdir(parents=True, exist_ok=True)
        # save config files
        with open(exp_folder / 'model.yaml', 'w') as f:
            yaml.safe_dump(self.config.model_config, f)
        with open(exp_folder / 'dls.yaml', 'w') as f:
            yaml.safe_dump(self.config.dls_config, f)
        with open(exp_folder / 'trainer.yaml', 'w') as f:
            yaml.safe_dump(self.config.trainer_config, f)
        # also create checkpoints folder
        (exp_folder / 'checkpoints').mkdir(parents=True, exist_ok=True)
        self.exp_folder = exp_folder

    def init_dls(self):
        dls_config = self.config.dls_config
        dls = {}
        for name in ['train', 'valid']:
            config = dls_config[name]
            class_name = config['ds_class']
            init_args = config['ds_init_args']
            import datasets
            config_cls = getattr(datasets, class_name + 'Config')
            cls = getattr(datasets, class_name)
            ds: datasets.base = cls(config_cls(**init_args)) # generic reference to dataset
            dls[name] = DataLoader(ds, collate_fn=ds.get_collate_function(), num_workers=8, pin_memory=True, drop_last=True, **config['dl_init_args'])
        self.dls = dls

    def init_model(self):
        model_config = self.config.model_config
        class_name = model_config['class']
        init_args = model_config['init_args']
        import models
        config_cls = getattr(models, class_name + 'Config')
        cls = getattr(models, class_name)
        self.model = cls(config_cls(**init_args))
        # compile model
        # TBD: compile on demand
        # import torch
        # self.model = torch.compile(self.model)

    def init_trainer(self):
        trainer_config = self.config.trainer_config
        # init callbacks
        self._experiment_stopper = ExperimentStopper(self._state)
        val_loss_ckpt = ModelCheckpoint(
            self.exp_folder / 'checkpoints/',
            filename='model-{epoch:02d}-{val_loss:.2f}',
            mode='min',
            monitor='val_loss',
            save_top_k=2)
        # init logger
        logger = TensorBoardLogger(self.exp_folder, name='', default_hp_metric=False, log_graph=False)
        self.trainer = Trainer(accelerator='gpu', devices=1,
                               callbacks=[self._experiment_stopper, val_loss_ckpt],
                               logger=logger,
                               **trainer_config)
    
    def get_dict_representation(self):
        ''' Returns a dictionary representation of the experiment. '''
        return {
            'name': self.name,
            'state': self.state,
            'config': {
                'model': self.config.model_config,
                'dl': self.config.dls_config,
                'trainer': self.config.trainer_config
            }
        }
    
    def __str__(self):
        return json.dumps(self.get_dict_representation(), indent=4)