from pathlib import Path
from itertools import product
from exp.manager import ExperimentManager, ExperimentConfig

exp_manager = ExperimentManager(Path('experiments/en-fr'))

#n_blocks = [4, 5, 6]
#attn_type = ['vanilla', 'multi_query', 'roformer_attn', 'xformer']
#max_len = [256, 512]
#use_spacy = [True, False]

n_blocks = [6]
attn_type = ['roformer_attn']
max_len = [512]
use_spacy = [False]

for n_block, attn, max_len, use_spacy in product(n_blocks, attn_type, max_len, use_spacy):
	exp_config_folder = Path('configs/en-fr-v1-spacy' if use_spacy else 'configs/en-fr-v1')
	config = ExperimentConfig.resuming_from_directory('experiments/en-fr/nb_6-at_roformer_attn-ml_512-bpe', checkpoint_name='model-epoch=00-val_loss=0.00.ckpt')
	config.model_config['init_args']['n_blocks'] = n_block
	config.model_config['init_args']['attention_type'] = attn
	config.model_config['init_args']['max_len'] = max_len
	config.dls_config['train']['ds_init_args']['max_seq_len'] = max_len
	config.dls_config['valid']['ds_init_args']['max_seq_len'] = max_len
	# test run only
	#config.dls_config['train']['ds_init_args']['first_n_lines'] = 32
	#config.dls_config['valid']['ds_init_args']['first_n_lines'] = 32
	exp_manager.create_and_append_experiment(f'nb_{n_block}-at_{attn}-ml_{max_len}-{"spacy" if use_spacy else "bpe"}', config)

input()