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
	sp_model_file: str
	src_file: str
	tgt_file: str

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
			src_lines = list(f)
		with open(config.tgt_file, encoding='utf8') as f:
			tgt_lines = list(f)
		self.df = pd.DataFrame({ 'src' : src_lines , 'tgt' : tgt_lines })
		self.tokenizer = sp.SentencePieceProcessor(model_file=config.sp_model_file)

	def __len__(self):
		return len(self.df)
	
	def __getitem__(self, idx):
		row = self.df.iloc[idx]
		src, tgt = row.src, row.tgt
		# enable bpe dropout regularization
		x = self.tokenizer.encode(src, enable_sampling=True, alpha=0.1, nbest_size=-1)
		y = self.tokenizer.encode(tgt, enable_sampling=True, alpha=0.1, nbest_size=-1)
		# create src & tgt tensors
		x_src = torch.tensor([TranslationDataset.BOS_IDX] + x + [TranslationDataset.EOS_IDX], dtype=torch.long)
		x_tgt = torch.tensor([TranslationDataset.BOS_IDX] + y[:-1], dtype=torch.long)
		y_tgt = torch.tensor(y[1:] + [TranslationDataset.EOS_IDX], dtype=torch.long)
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