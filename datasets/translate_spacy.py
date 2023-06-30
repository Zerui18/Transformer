import torch
import pandas as pd
import spacy

from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence
from pickle import load

from .base import BaseDataset
from models.transformer import TransformerInputBatch

### CONFIG ###

@dataclass
class TranslationDatasetSpacyConfig:
	src_model: str
	tgt_model: str
	src_vocab_file: str
	tgt_vocab_file: str
	src_file: str
	tgt_file: str
	max_seq_len: int
	first_n_lines: int = None

### DATASET ###

class TranslationDatasetSpacy(BaseDataset):

	UNK_IDX = 0
	BOS_IDX = 1
	EOS_IDX = 2
	PAD_IDX = 3

	def __init__(self, config: TranslationDatasetSpacyConfig):
		super().__init__()
		print('Reading input files...')
		with open(config.src_file, encoding='utf8') as f:
			if config.first_n_lines is None:
				src_lines = f.readlines()
			else:
				src_lines = [next(f) for _ in range(config.first_n_lines)]
		with open(config.tgt_file, encoding='utf8') as f:
			if config.first_n_lines is None:
				tgt_lines = f.readlines()
			else:
				tgt_lines = [next(f) for _ in range(config.first_n_lines)]
		self.max_seq_len = config.max_seq_len
		self.df = pd.DataFrame({ 'src' : src_lines , 'tgt' : tgt_lines })
		print('Initializing tokenizers...')
		self.src_tokenizer = spacy.load(config.src_model)
		self.tgt_tokenizer = spacy.load(config.tgt_model)
		print('Initializing vocabularies...')
		with open(config.src_vocab_file, 'rb') as f:
			self.src_vocab = load(f)
		with open(config.tgt_vocab_file, 'rb') as f:
			self.tgt_vocab = load(f)
		print('Dataset initialized.')

	def __len__(self):
		return len(self.df)
	
	def __getitem__(self, idx):
		row = self.df.iloc[idx]
		src, tgt = row.src, row.tgt
		# tokenize & convert to indices
		x = self.src_tokenizer(src[:self.max_seq_len])
		x = [self.src_vocab[str(t)] for t in x]
		y = self.tgt_tokenizer(tgt[:self.max_seq_len])
		y = [self.tgt_vocab[str(t)] for t in y]
		# create src & tgt tensors
		x_src = torch.tensor(x, dtype=torch.long)
		x_tgt = torch.tensor([TranslationDatasetSpacy.BOS_IDX] + y[:-1], dtype=torch.long)
		y_tgt = torch.tensor(y[1:] + [TranslationDatasetSpacy.EOS_IDX], dtype=torch.long)
		return x_src, x_tgt, y_tgt

	@staticmethod
	def make_pad_mask(x):
		return (x != TranslationDatasetSpacy.PAD_IDX) # (B, T)
	
	@staticmethod
	def get_collate_function() -> callable or None:
		def collate_function(batch):
			# batch is a list of tuples (src, tgt)
			x_src, x_tgt, y_tgt = zip(*batch)
			# convert src & tgt to tensors
			x_src = pad_sequence(x_src, batch_first=True, padding_value=TranslationDatasetSpacy.PAD_IDX)
			x_tgt = pad_sequence(x_tgt, batch_first=True, padding_value=TranslationDatasetSpacy.PAD_IDX)
			y_tgt = pad_sequence(y_tgt, batch_first=True, padding_value=TranslationDatasetSpacy.PAD_IDX)
			# create src_mask & tgt_mask
			x_src_tok_mask = TranslationDatasetSpacy.make_pad_mask(x_src)
			x_tgt_tok_mask = TranslationDatasetSpacy.make_pad_mask(x_tgt)
			return TransformerInputBatch(x_src, x_tgt, x_src_tok_mask, x_tgt_tok_mask, y_tgt)
		return collate_function