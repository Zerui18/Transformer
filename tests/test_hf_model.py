from pathlib import Path
from exp.manager import ExperimentManager, ExperimentConfig

exp_manager = ExperimentManager(Path('experiments/en-de-hf-spacy-multi30k'))

config = ExperimentConfig.from_config_files('configs/de-en-hf-spacy-mutli30k/model.yaml',
                                            'configs/de-en-hf-spacy-mutli30k/dls.yaml',
                                            'configs/de-en-hf-spacy-mutli30k/trainer.yaml')
exp_manager.create_and_append_experiment(f'en-de-hf-spacy-multi30k', config)

input()