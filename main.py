from argparse import ArgumentParser

def build_argparser():
	parser = ArgumentParser()
	subparsers = parser.add_subparsers(dest='command')
	# train arguments
	train = subparsers.add_parser('train')
	train.add_argument('--model-config', type=str, required=True)
	train.add_argument('--train-config', type=str, required=True)
	train.add_argument('--experiment-name', type=str, required=True)
	# train_resume arguments
	train_resume = subparsers.add_parser('train_resume')
	train_resume.add_argument('--experiment-name', type=str, required=True)
	# translate arguments
	translate = subparsers.add_parser('translate')
	translate.add_argument('--checkpoint-path', type=str, required=True)
	return parser

if __name__ == '__main__':
	parser = build_argparser()
	args = parser.parse_args()
	match args.command:
		case 'train':
			from train import train
			train(args)
		case 'train_resume':
			from train import train_resume
			train_resume(args)
		case 'translate':
			from translate import translate
			translate(args)
		case _: # None
			print('Please specify a command!')
			parser.print_help()