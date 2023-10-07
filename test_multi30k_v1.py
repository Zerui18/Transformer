from pathlib import Path
from itertools import product
from exp.manager import ExperimentManager, ExperimentConfig

exp_manager = ExperimentManager(Path('experiments/multi30k-v2'))

ATTN_TYPES = ['multi_query', 'roformer_attn', 'xformer']
N_BLOCKS = [4, 5, 6]
VOCAB_SIZE = [1000, 5000, 10000, 15000]

for n_blocks, vocab_size, attn_type in product(N_BLOCKS, VOCAB_SIZE, ATTN_TYPES):
	config = ExperimentConfig.from_config_files(f'configs/de-en-v1-sp-multi30k/model.yaml',
												f'configs/de-en-v1-sp-multi30k/dls.yaml',
												f'configs/de-en-v1-sp-multi30k/trainer.yaml')
	config.model_config['init_args']['n_blocks'] = n_blocks
	config.model_config['init_args']['src_vocab_size'] = vocab_size
	config.model_config['init_args']['tgt_vocab_size'] = vocab_size
	config.model_config['init_args']['attention_type'] = attn_type
	config.model_config['tokenizer']['init_args']['sp_model_path'] = f'data/multi30k/multi30k_{vocab_size}.model'
	config.dls_config['train']['ds_init_args']['src_sp_model_file'] = f'data/multi30k/multi30k_{vocab_size}.model'
	config.dls_config['train']['ds_init_args']['tgt_sp_model_file'] = f'data/multi30k/multi30k_{vocab_size}.model'
	config.dls_config['valid']['ds_init_args']['src_sp_model_file'] = f'data/multi30k/multi30k_{vocab_size}.model'
	config.dls_config['valid']['ds_init_args']['tgt_sp_model_file'] = f'data/multi30k/multi30k_{vocab_size}.model'
	exp_manager.create_and_append_experiment(f'multi30k-v2-nb_{n_blocks}-v_{vocab_size}-at_{attn_type}', config)

input()