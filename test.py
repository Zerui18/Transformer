from exp.manager import ExperimentManager, ExperimentConfig

exp_manager = ExperimentManager(single_process=False)

# config = ExperimentConfig.from_config_files('configs/en-de-v1/model.yaml', 'configs/en-de-v1/dls.yaml', 'configs/en-de-v1/trainer.yaml')
# exp_manager.create_and_append_experiment('sep_vocab', config)

# config2 = ExperimentConfig.from_config_files('configs/en-de-altdl/model.yaml', 'configs/en-de-altdl/dls.yaml', 'configs/en-de-altdl/trainer.yaml')
# exp_manager.create_and_append_experiment('alt_dl', config2)

attn_types = ['vanilla', 'stock', 'flash']
n_blocks = [3, 4, 5]

for attn_type in attn_types:
    for n_block in n_blocks:
        config = ExperimentConfig.from_config_files('configs/en-de-altdl/model.yaml', 'configs/en-de-altdl/dls.yaml', 'configs/en-de-altdl/trainer.yaml')
        config.model_config['attenion_type'] = attn_type
        config.model_config['init_args']['n_blocks'] = n_block
        # config.trainer_config['max_steps'] = 10
        exp_manager.create_and_append_experiment(f'{attn_type}_attn_{n_block}_blks', config)
    # input()
    # import pdb
    # pdb.set_trace()

# print('finalle')

input()
# import pdb
# pdb.set_trace()
