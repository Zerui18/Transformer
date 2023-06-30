from exp.manager import ExperimentManager, ExperimentConfig
from pathlib import Path

exp_manager = ExperimentManager(Path('experiments/en-fr'))

exp_config_folder = Path('configs/en-fr-v1')

config = ExperimentConfig.from_config_files(exp_config_folder / 'model.yaml', exp_config_folder / 'dls.yaml', exp_config_folder / 'trainer.yaml')
exp_manager.create_and_append_experiment('v1-sp', config)

input()