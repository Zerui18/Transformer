from exp.manager import ExperimentManager, ExperimentConfig

exp_manager = ExperimentManager()
n_blocks = [4, 5, 6]

for n_block in n_blocks:
    config = ExperimentConfig.from_config_files('configs/en-de-altdl-hf/model.yaml', 'configs/en-de-altdl-hf/dls-mine.yaml', 'configs/en-de-altdl-hf/trainer.yaml')
    config.model_config['init_args']['n_blocks'] = n_block
    exp_manager.create_and_append_experiment(f'mydl_hf_bert_{n_block}_blks', config)

input()