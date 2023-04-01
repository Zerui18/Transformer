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
    QUEUING = 0 # waiting to be run
    RUNNING = 1 # currently running
    STOPPED = 2 # stopped by user (will not be run until changed to queuing)
    COMPLETED = 3 # finished running

class ExperimentStopper(Callback):

    state: Value

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

    def __init__(self, model_config_file: str, dls_config_file: str, trainer_config_file: str):
        self.model_config_file = model_config_file
        self.dls_config_file = dls_config_file
        self.trainer_config_file = trainer_config_file
        self._load_config_files()
    
    def _load_config_files(self):
        with open(self.model_config_file, 'r') as f:
            self.model_config = yaml.safe_load(f)
        with open(self.dls_config_file, 'r') as f:
            self.dl_config = yaml.safe_load(f)
        with open(self.trainer_config_file, 'r') as f:
            self.trainer_config = yaml.safe_load(f)


class Experiment:

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
        self._state = state
        self.name = name
        self.config = config

    def start(self):
        self.state = ExperimentState.QUEUING
    
    def stop(self):
        self.state = ExperimentState.STOPPED
    
    def run(self):
        self.state = ExperimentState.RUNNING
        self.trainer.fit(self.model, self.dls['train'], self.dls['valid'])
        # only set completed if not stopped
        if self.state == ExperimentState.RUNNING:
            self.state = ExperimentState.COMPLETED

    def init_resources(self):
        ''' Initialize all resources needed for the experiment. 
        Note: This should only be called in the subprocess.
        '''
        self.init_exp_folder()
        self.init_dls()
        self.init_model()
        self.init_trainer()

    def init_exp_folder(self):
        exp_folder = Path(f'experiments/{self.name}')
        exp_folder.mkdir(parents=True, exist_ok=True)
        # copy config files
        for config_file in [self.config.dls_config_file, self.config.model_config_file, self.config.trainer_config_file]:
            shutil.copy(config_file, exp_folder)
        # also create checkpoints folder
        (exp_folder / 'checkpoints').mkdir(parents=True, exist_ok=True)
        self.exp_folder = exp_folder

    def init_dls(self):
        dl_config = self.config.dl_config
        dls = {}
        for name in ['train', 'valid']:
            config = dl_config[name]
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
                'dl': self.config.dl_config,
                'trainer': self.config.trainer_config
            }
        }
    
    def __str__(self):
        return json.dumps(self.get_dict_representation(), indent=4)