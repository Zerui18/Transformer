from pathlib import Path
from exp.manager import ExperimentManager, ExperimentConfig

exp_manager = ExperimentManager(Path('experiments/de-en-v1-multi30k'))

N_BLOCKS = [6]
TOKENIZERS = ['spacymy']

for n_blocks in N_BLOCKS:
	for tokenizer in TOKENIZERS:
		config = ExperimentConfig.from_config_files(f'configs/de-en-v1-{tokenizer}-mutli30k/model.yaml',
													f'configs/de-en-v1-{tokenizer}-mutli30k/dls.yaml',
													f'configs/de-en-v1-{tokenizer}-mutli30k/trainer.yaml')
		exp_manager.create_and_append_experiment(f'de-en-v1-{tokenizer}-nb_{n_blocks}-multi30k', config)

input()