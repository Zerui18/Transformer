from pathlib import Path
from exp.manager import ExperimentManager, ExperimentConfig

exp_manager = ExperimentManager(Path('experiments/en-fr-v1'))

config = ExperimentConfig.from_config_files(f'configs/en-fr-v1/model.yaml',
											f'configs/en-fr-v1/dls.yaml',
											f'configs/en-fr-v1/trainer.yaml')

exp_manager.create_and_append_experiment(f'en-fr-v1', config)

input()