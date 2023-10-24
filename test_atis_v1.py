from pathlib import Path
from itertools import product
from exp.manager import ExperimentManager, ExperimentConfig

exp_manager = ExperimentManager(Path('experiments/atis-v1'), single_process=True)

N_BLOCKS = [4, 5, 6]
VOCAB_SIZE = [1000, 2000, 3000]
N_CNN_LAYERS = [1, 3, 5]

for n_blocks, n_cnn_layers, vocab_size in product(N_BLOCKS, N_CNN_LAYERS, VOCAB_SIZE):
	config = ExperimentConfig.from_config_files(f'configs/atis-v1/model.yaml',
												f'configs/atis-v1/dls.yaml',
												f'configs/atis-v1/trainer.yaml')
	config.model_config['init_args']['n_blocks'] = n_blocks
	config.model_config['init_args']['n_cnn_layers'] = n_cnn_layers
	config.model_config['init_args']['vocab_size'] = vocab_size
	config.dls_config['train']['ds_init_args']['sp_model'] = f'data/atis/atis_{vocab_size}.model'
	config.dls_config['valid']['ds_init_args']['sp_model'] = f'data/atis/atis_{vocab_size}.model'
	exp_manager.create_and_append_experiment(f'atis-v1-nb_{n_blocks}-nc_{n_cnn_layers}-v_{vocab_size}', config)

input()