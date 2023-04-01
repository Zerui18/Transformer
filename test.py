from exp.manager import ExperimentManager, Experiment, ExperimentConfig

exp_manager = ExperimentManager()
config = ExperimentConfig('configs/en-de-altdl/model.yaml', 'configs/en-de-altdl/dls.yaml', 'configs/en-de-altdl/trainer.yaml')
exp_manager.create_and_append_experiment('test', config)

print('Current', exp_manager.current_experiment)
print('Queued', exp_manager.queued_experiments)

input()

exp_manager.stop_current_experiment()

print('Current', exp_manager.current_experiment)
print('Queued', exp_manager.queued_experiments)

input()

exp_manager.start_all()

print('Current', exp_manager.current_experiment)
print('Queued', exp_manager.queued_experiments)

input()