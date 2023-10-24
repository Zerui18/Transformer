from pathlib import Path
from itertools import product
from exp.manager import ExperimentManager, ExperimentConfig

exp_manager = ExperimentManager(Path('experiments/multi30k-v2-rp2'))

N_BLOCKS = [4]
N_HEADS = [16]
EMB_DIM = [512]
weight_tying = ['2-way']
vocab_size = 1000
attn_type = 'roformer_attn'

for n_blocks, n_heads, emb_dim, wt in product(N_BLOCKS, N_HEADS, EMB_DIM, weight_tying):
	for i in range(8):
		config = ExperimentConfig.from_config_files(f'configs/de-en-v1-sp-multi30k/model.yaml',
													f'configs/de-en-v1-sp-multi30k/dls.yaml',
													f'configs/de-en-v1-sp-multi30k/trainer.yaml')
		config.model_config['init_args']['n_blocks'] = n_blocks
		config.model_config['init_args']['src_vocab_size'] = vocab_size
		config.model_config['init_args']['tgt_vocab_size'] = vocab_size
		config.model_config['init_args']['attention_type'] = attn_type
		config.model_config['init_args']['n_heads'] = n_heads
		config.model_config['init_args']['emb_dim'] = emb_dim
		config.model_config['init_args']['weight_tying'] = wt
		config.model_config['tokenizer']['init_args']['sp_model_path'] = f'data/multi30k/multi30k_{vocab_size}.model'
		config.dls_config['train']['ds_init_args']['src_sp_model_file'] = f'data/multi30k/multi30k_{vocab_size}.model'
		config.dls_config['train']['ds_init_args']['tgt_sp_model_file'] = f'data/multi30k/multi30k_{vocab_size}.model'
		config.dls_config['valid']['ds_init_args']['src_sp_model_file'] = f'data/multi30k/multi30k_{vocab_size}.model'
		config.dls_config['valid']['ds_init_args']['tgt_sp_model_file'] = f'data/multi30k/multi30k_{vocab_size}.model'
		exp_manager.create_and_append_experiment(f'rp_{i}', config)

input()