import argparse
import numpy as np
import librosa
import os
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool

def parse_args():
	parser = argparse.ArgumentParser()
	# 3 modes: calc_mel, calc_mel_stats, norm_mel, each having a different set of arguments
	subparsers = parser.add_subparsers(dest='mode', required=True)
	# calc_mel
	parser_calc_mel = subparsers.add_parser('calc_mel')
	parser_calc_mel.add_argument('--data_dir', type=Path, required=True)
	parser_calc_mel.add_argument('--output_dir', type=Path, required=True)
	parser_calc_mel.add_argument('--sample_rate', type=int, default=16000)
	parser_calc_mel.add_argument('--n_fft', type=int, default=1024)
	parser_calc_mel.add_argument('--win_length', type=int, default=400)
	parser_calc_mel.add_argument('--hop_length', type=int, default=160)
	parser_calc_mel.add_argument('--n_mels', type=int, default=80)
	parser_calc_mel.add_argument('--fmin', type=int, default=0)
	parser_calc_mel.add_argument('--fmax', type=int, default=8000)
	parser_calc_mel.add_argument('--max_duration', type=float, default=10.0)
	# calc_mel_stats
	parser_calc_mel_stats = subparsers.add_parser('calc_mel_stats')
	parser_calc_mel_stats.add_argument('--data_dir', type=Path, required=True)
	parser_calc_mel_stats.add_argument('--output_file', type=Path, required=True)
	# norm_mel
	parser_norm_mel = subparsers.add_parser('norm_mel')
	parser_norm_mel.add_argument('--mel_dir', type=Path, required=True)
	parser_norm_mel.add_argument('--stats_file', type=Path, required=True)
	parser_norm_mel.add_argument('--output_dir', type=Path, required=True)
	return parser.parse_args()

### CALC MEL ###
def calc_mel_worker(args: dict):
	# obtain args
	audio_path = args['audio_path']
	output_dir = args['output_dir']
	sample_rate = args['sample_rate']
	n_fft = args['n_fft']
	win_length = args['win_length']
	hop_length = args['hop_length']
	n_mels = args['n_mels']
	fmin = args['fmin']
	fmax = args['fmax']
	max_duration = args['max_duration']
	audio, sr = librosa.load(audio_path, sr=sample_rate)
	# filter by duration
	if len(audio) > max_duration * sr:
		return
	# resample if needed
	if sr != sample_rate:
		audio = librosa.resample(audio, sr, sample_rate)
	# cal mel-spectrogram
	mel = librosa.feature.melspectrogram(
		y=audio,
		sr=sample_rate,
		n_fft=n_fft,
		win_length=win_length,
		hop_length=hop_length,
		n_mels=n_mels,
		fmin=fmin,
		fmax=fmax,
	)
	# write to file
	output_path = output_dir / (audio_path.stem + '.npy')
	np.save(output_path, mel)

def calc_mel(args):
	output_dir = args.output_dir
	output_dir.mkdir(parents=True, exist_ok=True)
	audio_paths = list(Path(args.data_dir).glob('*.wav'))
	worker_args = [dict(audio_path=audio_path,
		     output_dir=Path(args.output_dir),
			 sample_rate=args.sample_rate,
			 n_fft=args.n_fft,
			 win_length=args.win_length,
			 hop_length=args.hop_length,
			 n_mels=args.n_mels,
			 fmin=args.fmin,
			 fmax=args.fmax,
			 max_duration=args.max_duration) for audio_path in audio_paths]
	with Pool(processes=os.cpu_count()) as pool:
		for _ in tqdm(pool.imap_unordered(calc_mel_worker, worker_args), total=len(audio_paths)):
			pass

### CALC MEL STATS ###
def calc_mel_stats(args):
	data_dir = args.data_dir
	output_file = args.output_file
	# collect all mel-spectrograms
	mels = []
	for mel_path in tqdm(Path(data_dir).glob('*.npy')):
		mel = np.load(mel_path)
		mels.append(mel)
	# calculate global mean & var
	mels_1d = np.concatenate([mel.flatten() for mel in mels])
	mean = np.mean(mels_1d)
	var = np.var(mels_1d)
	abs_max = np.max(np.abs(mels_1d))
	# write to file
	with open(output_file, 'wb') as f:
		np.save(f, np.array([mean, var, abs_max]))

### NORM MEL ###
def norm_mel_worker(args: dict):
	# obtain args
	mel_path = args['mel_path']
	stats = args['stats']
	output_dir = args['output_dir']
	# load mel
	mel = np.load(mel_path)
	# normalize
	mel = (mel - stats[0]) / stats[2]
	# write to file
	output_path = output_dir / mel_path.name
	np.save(output_path, mel)

def norm_mel(args):
	mel_dir = args.mel_dir
	stats_file = args.stats_file
	output_dir = args.output_dir
	output_dir.mkdir(parents=True, exist_ok=True)
	# load stats
	with open(stats_file, 'rb') as f:
		stats = np.load(f)
	# normalize all mels
	mel_paths = list(Path(mel_dir).glob('*.npy'))
	worker_args = [dict(mel_path=mel_path, stats=stats, output_dir=output_dir) for mel_path in mel_paths]
	with Pool(processes=os.cpu_count()) as pool:
		for _ in tqdm(pool.imap_unordered(norm_mel_worker, worker_args), total=len(mel_paths)):
			pass

def main():
	args = parse_args()
	if args.mode == 'calc_mel':
		calc_mel(args)
	elif args.mode == 'calc_mel_stats':
		calc_mel_stats(args)
	elif args.mode == 'norm_mel':
		norm_mel(args)
	else:
		raise ValueError(f'Invalid mode: {args.mode}')
	
if __name__ == '__main__':
	main()