from exp.manager import ExperimentManager, ExperimentConfig

exp_manager = ExperimentManager()
attn_types = ['roformer_attn']
n_blocks = [4, 5]

for attn_type in attn_types:
    for n_block in n_blocks:
        config = ExperimentConfig.from_config_files('configs/en-de-altdl/model.yaml', 'configs/en-de-altdl/dls.yaml', 'configs/en-de-altdl/trainer.yaml')
        config.model_config['attenion_type'] = attn_type
        config.model_config['init_args']['n_blocks'] = n_block
        exp_manager.create_and_append_experiment(f'{attn_type}_attn_{n_block}_blks', config)

input()