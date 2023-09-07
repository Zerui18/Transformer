import torch
import pandas as pd
import math
import numpy as np
import sentencepiece as sp

from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence
from pickle import load
from pathlib import Path

from .base import BaseDataset
from models.whisper import WhisperInputBatch

### CONFIG ###

@dataclass
class ATISDatasetConfig:
	sp_model: str
	mel_dir: str
	transcripts_file: str
	dec_max_len: int
	first_n_lines: int = None

### DATASET ###

class ATISDataset(BaseDataset):

	UNK_IDX = 0
	BOS_IDX = 1
	EOS_IDX = 2
	PAD_IDX = 3

	def __init__(self, config: ATISDatasetConfig):
		super().__init__()
		self.max_seq_len = config.dec_max_len
		print('DS INIT:', config)
		# read transcripts
		with open(config.transcripts_file, encoding='utf8') as f:
			file_ids, transcripts = zip(*[(line[:8], line[9:]) for line in f.readlines() if len(line.strip()) > 0])
		df = pd.DataFrame({ 'file_id' : file_ids , 'transcript' : transcripts })
		# read mels
		mel_dir = Path(config.mel_dir)
		mel_paths = [mel_dir / f'{file_id}.npy' for file_id in file_ids]
		mels = [np.load(mel_path) if mel_path.exists() else None for mel_path in mel_paths]
		# add mels to df
		df['mel'] = mels
		# filter out mels that are None
		df = df[df.mel.notnull()]
		# init tokenizer
		self.tokenizer = sp.SentencePieceProcessor(model_file=config.sp_model)
		# print stats
		print('DS HEAD:')
		print(df.head())
		self.df = df
		print('DS INIT COMPLETE')

	def __len__(self):
		return len(self.df)
	
	def __getitem__(self, idx):
		row = self.df.iloc[idx]
		mel, transcript = row.mel, row.transcript
		# enable bpe dropout regularization
		transcript = self.tokenizer.encode(transcript, enable_sampling=True, alpha=0.1, nbest_size=-1)[:self.max_seq_len]
		# create src & tgt tensors
		x_src = torch.tensor(mel.T, dtype=torch.float)
		x_tgt = torch.tensor([ATISDataset.BOS_IDX] + transcript[:-1], dtype=torch.long)
		y_tgt = torch.tensor(transcript[1:] + [ATISDataset.EOS_IDX], dtype=torch.long)
		return x_src, x_tgt, y_tgt

	@staticmethod
	def make_pad_mask(x: torch.LongTensor):
		return (x != ATISDataset.PAD_IDX) # (B, T)
	
	@staticmethod
	def make_pad_mask_with_lengths(max_len: int, lengths: torch.LongTensor):
		return torch.arange(max_len)[None, :] < lengths[:, None]
	
	@staticmethod
	def get_collate_function() -> callable or None:
		def collate_function(batch):
			# batch is a list of tuples (src, tgt)
			x_src, x_tgt, y_tgt = zip(*batch)
			# convert src & tgt to tensors
			lengths = torch.tensor([len(x) for x in x_src], dtype=torch.long)
			x_src = pad_sequence(x_src, batch_first=True, padding_value=0.0)
			x_tgt = pad_sequence(x_tgt, batch_first=True, padding_value=ATISDataset.PAD_IDX)
			y_tgt = pad_sequence(y_tgt, batch_first=True, padding_value=ATISDataset.PAD_IDX)
			# create src_mask & tgt_mask
			x_src_tok_mask = ATISDataset.make_pad_mask_with_lengths(math.ceil(x_src.shape[1] / 2), torch.ceil(lengths // 2))
			x_tgt_tok_mask = ATISDataset.make_pad_mask(x_tgt)
			return WhisperInputBatch(x_src, x_tgt, x_src_tok_mask, x_tgt_tok_mask, y_tgt)
		return collate_function