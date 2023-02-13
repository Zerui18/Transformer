import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from argparse import ArgumentParser
import yaml
from model_pl import TransformerModelLN
from dataset import TranslationDataset
from config import *
from pathlib import Path

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')
pl.seed_everything(2023, workers=True)

def build_argparser():
	parser = ArgumentParser()
	subparsers = parser.add_subparsers(dest='command')
	# train arguments
	train = subparsers.add_parser('train')
	train.add_argument('--model-config', type=str, required=True)
	train.add_argument('--train-config', type=str, required=True)
	train.add_argument('--experiment-name', type=str, required=True)
	# translate arguments
	translate = subparsers.add_parser('translate')
	return parser

def train(args):
	# load configs
	with open(args.model_config) as f:
		model_config = TransformerConfig(**yaml.safe_load(f))
	with open(args.train_config) as f:
		train_config = TrainingConfig(**yaml.safe_load(f))
	# init dataloaders
	print('Init dataloaders...')
	ds_train = TranslationDataset(train_config.train_src_file, train_config.train_dst_file, train_config.sp_model, model_config.block_size)
	ds_val   = TranslationDataset(train_config.val_src_file, train_config.val_dst_file, train_config.sp_model, model_config.block_size)
	dl_train = DataLoader(ds_train, train_config.batch_size, train_config.shuffle, num_workers=8, pin_memory=True, drop_last=True)
	dl_val   = DataLoader(ds_val, train_config.batch_size, train_config.shuffle, num_workers=8, pin_memory=True, drop_last=True)
	# auto-tuning
	if train_config.autotune_batch_size:
		print('Tuning batch size...')
		model = TransformerModelLN(model_config, train_config)
		trainer = pl.Trainer(accelerator='gpu', devices=1, auto_scale_batch_size=True)
		trainer.tune(model, dl_train, dl_val)
		batch_size = model.batch_size
		print('Found recommended batch size = ', batch_size)
		train_config.batch_size = batch_size
		del model, trainer, dl_train, dl_val
		# reinit the dataloaders with the new batch size
		dl_train = DataLoader(ds_train, batch_size, train_config.shuffle, num_workers=8, pin_memory=True, drop_last=True)
		dl_val   = DataLoader(ds_val, batch_size, train_config.shuffle, num_workers=8, pin_memory=True, drop_last=True)
	if train_config.autotune_learning_rate:
		print('Tuning learning rate...')
		model = TransformerModelLN(model_config, train_config)
		trainer = pl.Trainer(accelerator='gpu', devices=1, auto_lr_find=True)
		trainer.tune(model, dl_train, dl_val)
		lr = model.learning_rate
		print('Found recommended lr = ', lr)
		train_config.learning_rate = lr
		del model, trainer
	# summary
	print('***' * 20)
	print('Run Summary for', args.experiment_name)
	print('***' * 20)
	print('Model Config')
	print(model_config)
	print('Train Config')
	print(train_config)
	print('***' * 20)
	# create experiment folder & save configs
	experiments = Path('experiments')
	experiments.mkdir(exist_ok=True)
	exp_folder: Path = experiments / args.experiment_name
	exp_folder.mkdir()
	with open(exp_folder / 'train.yaml', 'w') as f:
		yaml.dump(train_config, f)
	with open(exp_folder / 'model.yaml', 'w') as f:
		yaml.dump(model_config, f)
	# init trainer
	print('Begin training...')
	val_loss_ckpt = ModelCheckpoint(
		exp_folder, 
		filename='model-{epoch:02d}-{val_loss:.2f}',
		mode='min',
		monitor='val_loss',
		save_top_k=2)
	trainer = pl.Trainer(accelerator='gpu', devices=1,
			  max_steps=train_config.steps,
			  accumulate_grad_batches=train_config.gradient_accum_steps,
			  callbacks=[val_loss_ckpt])
	trainer.fit(model, dl_train, dl_val)
	trainer.save_checkpoint(exp_folder / 'model-final.pt')

def translate(args):
	pass

if __name__ == '__main__':
	args = build_argparser().parse_args()
	match args.command:
		case 'train':
			train(args)
		case 'translate':
			translate(args)