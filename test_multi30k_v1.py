from pathlib import Path
from itertools import product

from exp.manager import ExperimentManager
from exp.experiment import ExperimentConfig

exp_manager = ExperimentManager(Path('experiments/multi30k-v2'))

ATTN_TYPES = ['roformer']
N_BLOCKS = [6]
VOCAB_SIZE = [10000]

for n_blocks, vocab_size, attn_type in product(N_BLOCKS, VOCAB_SIZE, ATTN_TYPES):
	config = ExperimentConfig.from_config_files(
		'experiments/de-en-v1-sp-multi30k/config/model.yaml',
		'experiments/de-en-v1-sp-multi30k/config/dataset.yaml',
		'experiments/de-en-v1-sp-multi30k/config/training.yaml')
	# override model hparams for grid search
	config.model_config['n_blocks'] = n_blocks
	config.model_config['src_vocab_size'] = vocab_size
	config.model_config['tgt_vocab_size'] = vocab_size
	config.model_config['attention_type'] = attn_type
	config.model_config['tokenizer']['sp_model_path'] = f'data/multi30k/multi30k_{vocab_size}.model'
	# override dataset paths
	for split in ['train', 'valid']:
		config.dataset_config[split]['src_sp_model_file'] = f'data/multi30k/multi30k_{vocab_size}.model'
		config.dataset_config[split]['tgt_sp_model_file'] = f'data/multi30k/multi30k_{vocab_size}.model'
	exp_manager.create_and_append_experiment(
		f'multi30k-v2-nb_{n_blocks}-v_{vocab_size}-at_{attn_type}', config)

input()
