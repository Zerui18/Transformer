from exp.manager import ExperimentManager, ExperimentConfig

exp_manager = ExperimentManager(single_process=True)
attn_types = ['meme', 'xformer', 'multi_query']
n_blocks = [4]
lrs = [1e-4]

for attn_type in attn_types:
    for n_block in n_blocks:
        for lr in lrs:
            config = ExperimentConfig.from_config_files('configs/en-de-altdl/model.yaml', 'configs/en-de-altdl/dls.yaml', 'configs/en-de-altdl/trainer.yaml')
            config.model_config['init_args']['attention_type'] = attn_type
            config.model_config['init_args']['n_blocks'] = n_block
            config.model_config['init_args']['learning_rate'] = lr
            exp_manager.create_and_append_experiment(f'{lr}_lr_{attn_type}_attn_{n_block}_blks', config)

input()