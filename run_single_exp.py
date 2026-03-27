from pathlib import Path
from argparse import ArgumentParser

from exp.manager import ExperimentManager
from exp.experiment import ExperimentConfig


def parse_args():
	parser = ArgumentParser()
	parser.add_argument('--exp-name', type=Path, required=True, help='Name of experiment')
	parser.add_argument('--exp-save-path', type=Path, required=True, help='Path to save experiment')
	parser.add_argument('--model-config-path', type=Path, required=True, help='Path to model config')
	parser.add_argument('--dataset-config-path', type=Path, required=True, help='Path to dataset config')
	parser.add_argument('--training-config-path', type=Path, required=True, help='Path to training config')
	return parser.parse_args()

def main():
	args = parse_args()
	exp_manager = ExperimentManager(args.exp_save_path, single_process=True)
	config = ExperimentConfig.from_config_files(args.model_config_path, args.dataset_config_path, args.training_config_path)
	exp_manager.create_and_append_experiment(args.exp_name, config)

if __name__ == '__main__':
	main()