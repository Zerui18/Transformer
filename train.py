import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path
import yaml
from model import TransformerModelLN, TransformerModelStockLN
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
	# if already resuming from a checkpoint, return the experiment folder
	if RESUMING_FROM_CKPT:
		return exp_folder
	# else check if experiment folder already exists
	if exp_folder.exists():
		print('Experiment folder already exists!')
		choose_resume_ckpt(exp_folder)
	else:
		exp_folder.mkdir()
	return exp_folder

def choose_resume_ckpt(exp_folder: Path):
	global RESUMING_FROM_CKPT
	checkpoints = list(exp_folder.glob('model-*.ckpt'))
	if len(checkpoints) > 0:
		# print existing checkpoints and resume from chosen checkpoint
		print('Existing checkpoints:')
		for i, ckpt in enumerate(checkpoints):
			print(f'{i}: {ckpt.name}')
		choice = int(input('Enter checkpoint number to resume from: '))
		ckpt = checkpoints[choice]
		RESUMING_FROM_CKPT = ckpt
		return ckpt
	else:
		return None

def init_trainer(train_config: TrainingConfig, exp_folder: Path):
	# init callbacks
	val_loss_ckpt = ModelCheckpoint(
		exp_folder,
		filename='model-{epoch:02d}-{val_loss:.2f}',
		mode='min',
		monitor='val_loss',
		save_top_k=2)
	# init logger
	logger = TensorBoardLogger(exp_folder, name='tb_logs', default_hp_metric=False, log_graph=False)
	# init trainer
	trainer = pl.Trainer(accelerator='gpu', devices=1,
			  max_steps=train_config.max_steps,
			  accumulate_grad_batches=train_config.gradient_accum_steps,
			  gradient_clip_val=1,
			  callbacks=[val_loss_ckpt],
			  log_every_n_steps=1,
			  logger=logger,
			  track_grad_norm=True)
	return trainer

def train_resume(args):
	experiments = Path('experiments')
	experiments.mkdir(exist_ok=True)
	exp_folder = experiments / args.experiment_name
	if ckpt := choose_resume_ckpt(exp_folder):
		print(f'Resuming from {ckpt.name}...')
		args.model_config = ckpt.parent / 'model.yaml'
		args.train_config = ckpt.parent / 'train.yaml'
		train(args)
	else:
		print('No checkpoints found!')

def train(args):
	# load configs
	with open(args.model_config) as f:
		model_config = yaml.load(f, Loader=yaml.Loader)
		if type(model_config) is dict:
			model_config = TransformerConfig(**model_config)
	with open(args.train_config) as f:
		train_config = yaml.load(f, Loader=yaml.Loader)
		if type(train_config) is dict:
			train_config = TrainingConfig(**train_config)
	# init dataloaders
	dl_train, dl_val = init_dataloaders(train_config, model_config.block_size)
	# init model
	model = TransformerModelStockLN(model_config, train_config)
	model.train()
	# ensure exp_folder
	exp_folder = init_experiment(args.experiment_name)
	# init trainer
	trainer = init_trainer(train_config, exp_folder)
	# autotune lr (doesn't work with compile=True currently)
	if(RESUMING_FROM_CKPT is None) and train_config.autotune_learning_rate:
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