from __future__ import annotations

import argparse
import math
import random
import shutil
import time
from multiprocessing import Array, Process, Value
from pathlib import Path

from . import manager as manager_module
from .experiment import Experiment, ExperimentConfig, ExperimentState
from .server import serve
from .tb_events import ScalarEventWriter

# Demo / development driver for the web UI.
#
# Runs the REAL ExperimentManager + REST server, but swaps the child-process
# entrypoint for a fake trainer that sleeps through epochs, streams metrics into
# genuine TensorBoard event files, reports progress, honours stop requests, and
# can be scripted to fail — so every UI flow can be exercised live on a machine
# with no torch install, no GPU, and no datasets.
#
#   uv run python -m exp.demo               # http://127.0.0.1:5001/
#   uv run python -m exp.demo --keep        # keep previous demo folders (populates the disk panel)

def _write_err(err_buffer: Array, message: str):
	''' Writes an error message into a shared error buffer (mirrors Experiment.err_buffer).

	Args:
		1. err_buffer: Array  [bytes] the shared buffer
		2. message: str  the error text
	'''
	raw = message.encode('utf-8')[:len(err_buffer) - 1]
	err_buffer[:] = b'\x00' * len(err_buffer)
	err_buffer[:len(raw)] = raw

def _fake_child(name: str, directory: str, state: Value, err_buffer: Array, progress: Array, config: ExperimentConfig):
	''' Child-process entrypoint that impersonates a training run.

	Args:
		1. name: str  the experiment name (seeds the fake curves)
		2. directory: str  the experiment folder path
		3. state: Value  [int] shared ExperimentState
		4. err_buffer: Array  [bytes] shared error buffer
		5. progress: Array  [int32, (4,)] shared [epoch, batch, batches_per_epoch, global_step]
		6. config: ExperimentConfig  pacing is read from config.trainer_config['_demo']

	Recognised `_demo` keys: batches_per_epoch (int), seconds_per_batch (float),
	init_seconds (float), fail_at_epoch (int | None), fail_message (str).
	'''
	demo = config.trainer_config.get('_demo', {})
	max_epochs = int(config.trainer_config.get('max_epochs', 5))
	batches = int(demo.get('batches_per_epoch', 30))
	dt = float(demo.get('seconds_per_batch', 0.1))
	rng = random.Random(name)

	directory = Path(directory)
	# mirror Experiment._init_exp_folder so the folder looks like a real run
	directory.mkdir(parents=True, exist_ok=True)
	(directory / 'checkpoints').mkdir(exist_ok=True)
	import yaml
	for cfg_name, cfg in [('model', config.model_config), ('dls', config.dls_config), ('trainer', config.trainer_config)]:
		with open(directory / f'{cfg_name}.yaml', 'w') as f:
			yaml.safe_dump(cfg, f)

	time.sleep(float(demo.get('init_seconds', 1.0))) # pretend to build dataloaders/model
	with state.get_lock(): # compare-and-set: a stop requested during init must stick
		if state.value == ExperimentState.STOPPED:
			return
		state.value = ExperimentState.RUNNING

	writer = ScalarEventWriter(directory / 'version_0')
	base_loss = rng.uniform(3.0, 4.5)
	decay = batches * max_epochs / rng.uniform(2.0, 3.5)
	global_step = 0
	for epoch in range(max_epochs):
		for batch in range(batches):
			if state.value == ExperimentState.STOPPED:
				writer.close()
				return
			time.sleep(dt)
			global_step += 1
			loss = base_loss * math.exp(-global_step / decay) + 0.6 + rng.gauss(0, 0.05)
			writer.add_scalar('train_loss', loss, global_step)
			progress[:] = [epoch, batch + 1, batches, global_step]
		if demo.get('fail_at_epoch') == epoch:
			_write_err(err_buffer, demo.get('fail_message', 'RuntimeError: CUDA out of memory. Tried to allocate 2.31 GiB.'))
			state.value = ExperimentState.FAILED
			writer.close()
			return
		val_loss = base_loss * math.exp(-global_step / decay) + 0.7 + rng.gauss(0, 0.03)
		bleu = 38.0 * (1.0 - math.exp(-(epoch + 1) / (max_epochs / 2.5))) + rng.gauss(0, 0.5)
		writer.add_scalar('val_loss', val_loss, global_step)
		writer.add_scalar('bleu_greedy', max(0.0, bleu), global_step)
		writer.add_scalar('bleu_bs16', max(0.0, bleu + rng.uniform(0.5, 1.5)), global_step)
		# fake checkpoint files so the checkpoints panel has content
		(directory / 'checkpoints' / f'model-epoch={epoch}-step={global_step}-val_loss={val_loss:.2f}.ckpt').write_bytes(b'demo checkpoint\n')
	writer.close()
	with state.get_lock(): # compare-and-set: don't overwrite a last-moment stop request
		if state.value == ExperimentState.RUNNING:
			state.value = ExperimentState.COMPLETED

def _fake_run_experiment(experiment: Experiment) -> Process:
	''' Drop-in replacement for manager.run_experiment that launches the fake child.

	Args:
		1. experiment: Experiment  the experiment to launch
	Returns: process: Process  the started child process
	'''
	process = Process(target=_fake_child,
						args=(experiment.name, str(experiment.directory), experiment._state,
							experiment._err_buffer, experiment._progress, experiment.config))
	process.start()
	return process

def _fake_config(attention_type: str, n_blocks: int, max_epochs: int, seconds_per_batch: float,
					batches_per_epoch: int = 30, fail_at_epoch: int | None = None) -> ExperimentConfig:
	''' Builds a realistic-looking ExperimentConfig whose pacing is controlled via trainer_config['_demo'].

	Args:
		1. attention_type: str  fake model attention variant (display only)
		2. n_blocks: int  fake model depth (display only)
		3. max_epochs: int  epochs the fake child will run
		4. seconds_per_batch: float  sleep per fake batch
		5. batches_per_epoch: int  [30] fake batches per epoch
		6. fail_at_epoch: int | None  [None] epoch at which the fake child fails
	Returns: config: ExperimentConfig  the fake config
	'''
	model = {
		'class': 'Transformer',
		'init_args': {
			'n_blocks': n_blocks, 'n_heads': 8, 'emb_dim': 512, 'ff_dim': 2048,
			'dropout': 0.1, 'attention_type': attention_type,
			'src_vocab_size': 10000, 'tgt_vocab_size': 10000,
		},
		'tokenizer': {'class': 'SPTokenizer', 'init_args': {'sp_model_path': 'data/multi30k/multi30k_10000.model'}},
	}
	dls = {
		split: {
			'ds_class': 'TranslationDataset',
			'ds_init_args': {'split': split, 'src_lang': 'de', 'tgt_lang': 'en',
							'src_sp_model_file': 'data/multi30k/multi30k_10000.model',
							'tgt_sp_model_file': 'data/multi30k/multi30k_10000.model'},
			'dl_init_args': {'batch_size': 128, 'shuffle': split == 'train'},
		}
		for split in ['train', 'valid']
	}
	trainer = {
		'max_epochs': max_epochs,
		'gradient_clip_val': 1.0,
		'_demo': {'batches_per_epoch': batches_per_epoch, 'seconds_per_batch': seconds_per_batch,
					'fail_at_epoch': fail_at_epoch},
	}
	return ExperimentConfig(dls, model, trainer)

def main():
	''' Runs the demo: an ExperimentManager over fake experiments plus the web UI server. '''
	parser = argparse.ArgumentParser(description='zlab experiment manager web UI demo (fake experiments)')
	parser.add_argument('--master-dir', default='experiments/_demo', help='Directory for demo experiment folders')
	parser.add_argument('--host', default='127.0.0.1')
	parser.add_argument('--port', type=int, default=5001)
	parser.add_argument('--keep', action='store_true', help='Keep folders from a previous demo run (they appear in the disk panel)')
	args = parser.parse_args()

	master = Path(args.master_dir)
	if not args.keep and master.is_dir():
		shutil.rmtree(master)
	master.mkdir(parents=True, exist_ok=True)

	# swap the real (torch) child for the fake one BEFORE any experiment is enqueued
	manager_module.run_experiment = _fake_run_experiment
	exp_manager = manager_module.ExperimentManager(str(master))

	stamp = time.strftime('%H%M%S')
	specs = [
		(f'demo-{stamp}-vanilla-nb6', _fake_config('Vanilla', 6, max_epochs=6, seconds_per_batch=0.25)),
		(f'demo-{stamp}-roformer-nb6', _fake_config('RoFormer', 6, max_epochs=5, seconds_per_batch=0.2)),
		(f'demo-{stamp}-multiquery-nb4', _fake_config('MultiQuery', 4, max_epochs=4, seconds_per_batch=0.15)),
		(f'demo-{stamp}-flash-oom', _fake_config('Flash', 8, max_epochs=6, seconds_per_batch=0.1, fail_at_epoch=1)),
		(f'demo-{stamp}-softmax1-nb5', _fake_config('Softmax1', 5, max_epochs=5, seconds_per_batch=0.2)),
		(f'demo-{stamp}-average-nb4', _fake_config('Average', 4, max_epochs=4, seconds_per_batch=0.3)),
	]
	for exp_name, config in specs:
		exp_manager.create_and_append_experiment(exp_name, config)

	serve(exp_manager, host=args.host, port=args.port, block=True)

if __name__ == '__main__':
	main()
