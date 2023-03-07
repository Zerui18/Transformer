import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path
import yaml
from model_ln import TransformerModelLN
from dataset import TranslationDataset, TranslationBatch
from config import TransformerConfig, TrainingConfig

GPU = torch.device('cuda')
RESUMING_FROM_CKPT = None

torch.autograd.set_detect_anomaly(True) # detect backward NaN
torch.set_float32_matmul_precision('high') # allow tf32
pl.seed_everything(2023, workers=True) # static seed

def init_dataloaders(train_config: TrainingConfig, block_size: int):
    print('Init dataloaders...')
    ds_train = TranslationDataset(train_config.train_src_file, train_config.train_tgt_file, train_config.sp_model, block_size)
    ds_val   = TranslationDataset(train_config.val_src_file, train_config.val_tgt_file, train_config.sp_model, block_size)
    dl_train = DataLoader(ds_train, train_config.batch_size, train_config.shuffle, collate_fn=TranslationBatch.make_batch, num_workers=8, pin_memory=True, drop_last=True)
    dl_val   = DataLoader(ds_val, train_config.batch_size, collate_fn=TranslationBatch.make_batch, num_workers=8, pin_memory=True, drop_last=True)
    return dl_train, dl_val

def init_experiment(exp_name: str):
	global RESUMING_FROM_CKPT
	experiments = Path('experiments')
	experiments.mkdir(exist_ok=True)
	exp_folder: Path = experiments / exp_name
	if exp_folder.exists():
		print('Experiment folder already exists!')
		checkpoints = list(exp_folder.glob('model-*.ckpt'))
		if len(checkpoints) > 0:
			# print existing checkpoints and resume from chosen checkpoint
			print('Existing checkpoints:')
			for i, ckpt in enumerate(checkpoints):
				print(f'{i}: {ckpt.name}')
			choice = int(input('Enter checkpoint number to resume from: '))
			ckpt = checkpoints[choice]
			RESUMING_FROM_CKPT = ckpt
			print(f'Resuming from {ckpt.name}...')
	else:
		exp_folder.mkdir()
	return exp_folder

def init_trainer(train_config: TrainingConfig, exp_folder: Path):
	val_loss_ckpt = ModelCheckpoint(
		exp_folder,
		filename='model-{epoch:02d}-{val_loss:.2f}',
		mode='min',
		monitor='val_loss',
		save_top_k=2)
	trainer = pl.Trainer(accelerator='gpu', devices=1,
			  max_steps=train_config.max_steps,
			  accumulate_grad_batches=train_config.gradient_accum_steps,
			  gradient_clip_val=5,
			  callbacks=[val_loss_ckpt])
	return trainer

def train(args):
	# load configs
	with open(args.model_config) as f:
		model_config = TransformerConfig(**yaml.safe_load(f))
	with open(args.train_config) as f:
		train_config = TrainingConfig(**yaml.safe_load(f))
	# init dataloaders
	dl_train, dl_val = init_dataloaders(train_config, model_config.block_size)
	# init model
	model = TransformerModelLN(model_config, train_config)
	# ensure exp_folder
	exp_folder = init_experiment(args.experiment_name)
	# init trainer
	trainer = init_trainer(train_config, exp_folder)
	# autotune lr (doesn't work with compile=True currently)
	if not RESUMING_FROM_CKPT and train_config.autotune_learning_rate:
		print('Tuning learning rate...')
		trainer.tune(model, dl_train, dl_val)
		lr = model.learning_rate
		print('Found recommended lr = ', lr)
		train_config.learning_rate = lr
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
	with open(exp_folder / 'train.yaml', 'w') as f:
		yaml.dump(train_config, f)
	with open(exp_folder / 'model.yaml', 'w') as f:
		yaml.dump(model_config, f)
	# init trainer
	print('Begin training...')
	trainer.fit(model, dl_train, dl_val, ckpt_path=RESUMING_FROM_CKPT)
	trainer.save_checkpoint(exp_folder / 'model-final.pt')