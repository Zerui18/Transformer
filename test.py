from exp.manager import ExperimentManager, ExperimentConfig
from copy import deepcopy

exp_manager = ExperimentManager()

config = ExperimentConfig.from_config_files('configs/en-de-v1/model.yaml', 'configs/en-de-v1/dls.yaml', 'configs/en-de-v1/trainer.yaml')
exp_manager.create_and_append_experiment('sep_vocab', config)

config2 = deepcopy(config)
config2.model_config['optimizer'] = 'Adam'
exp_manager.create_and_append_experiment('sep_vocab_adam', config2)

print('Current', exp_manager.current_experiment)
print('Queued', exp_manager.queued_experiments)

input()

exp_manager.stop_current_experiment()

input()