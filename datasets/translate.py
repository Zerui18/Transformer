import torch
from torch.nn.utils.rnn import pad_sequence
import sentencepiece as sp
import pandas as pd
from dataclasses import dataclass
from .base import BaseDataset
from models.transformer import TransformerInputBatch

### CONFIG ###

@dataclass
class TranslationDatasetConfig:
	src_sp_model_file: str
	tgt_sp_model_file: str
	src_file: str
	tgt_file: str
	max_seq_len: int
	first_n_lines: int = None

### DATASET ###

class TranslationDataset(BaseDataset):

	UNK_IDX = 0
	BOS_IDX = 1
	EOS_IDX = 2
	PAD_IDX = 3

	def __init__(self, config: TranslationDatasetConfig):
		super().__init__()
		print('Reading input files...')
		with open(config.src_file, encoding='utf8') as f:
			if config.first_n_lines is None:
				src_lines = [l.strip() for l in f]
			else:
				src_lines = [next(f).strip() for _ in range(config.first_n_lines)]
		with open(config.tgt_file, encoding='utf8') as f:
			if config.first_n_lines is None:
				tgt_lines = [l.strip() for l in f]
			else:
				tgt_lines = [next(f).strip() for _ in range(config.first_n_lines)]
		self.max_seq_len = config.max_seq_len
		self.df = pd.DataFrame({ 'src' : src_lines , 'tgt' : tgt_lines })
		print(self.df.head(10))
		print('Loading sentencepiece models...')
		self.src_tokenizer = sp.SentencePieceProcessor(model_file=config.src_sp_model_file)
		self.tgt_tokenizer = sp.SentencePieceProcessor(model_file=config.tgt_sp_model_file)
		print('Done.')

	def __len__(self):
		return len(self.df)
	
	def __getitem__(self, idx):
		row = self.df.iloc[idx]
		src, tgt = row.src, row.tgt
		# enable bpe dropout regularization
		x = self.src_tokenizer.encode(src, enable_sampling=True, alpha=0.1, nbest_size=-1)[:self.max_seq_len]
		y = self.tgt_tokenizer.encode(tgt, enable_sampling=True, alpha=0.1, nbest_size=-1)[:self.max_seq_len]
		# create src & tgt tensors
		x_src = torch.tensor([TranslationDataset.BOS_IDX] + x + [TranslationDataset.EOS_IDX], dtype=torch.long)
		x_tgt = torch.tensor([TranslationDataset.BOS_IDX] + y, dtype=torch.long)
		y_tgt = torch.tensor(y + [TranslationDataset.EOS_IDX], dtype=torch.long)
		return x_src, x_tgt, y_tgt

	@staticmethod
	def make_pad_mask(x):
		return (x != TranslationDataset.PAD_IDX) # (B, T)
	
	@staticmethod
	def get_collate_function() -> callable or None:
		def collate_function(batch):
			# batch is a list of tuples (src, tgt)
			x_src, x_tgt, y_tgt = zip(*batch)
			# convert src & tgt to tensors
			x_src = pad_sequence(x_src, batch_first=True, padding_value=TranslationDataset.PAD_IDX)
			x_tgt = pad_sequence(x_tgt, batch_first=True, padding_value=TranslationDataset.PAD_IDX)
			y_tgt = pad_sequence(y_tgt, batch_first=True, padding_value=TranslationDataset.PAD_IDX)
			# create src_mask & tgt_mask
			x_src_tok_mask = TranslationDataset.make_pad_mask(x_src)
			x_tgt_tok_mask = TranslationDataset.make_pad_mask(x_tgt)
			return TransformerInputBatch(x_src, x_tgt, x_src_tok_mask, x_tgt_tok_mask, y_tgt)
		return collate_function