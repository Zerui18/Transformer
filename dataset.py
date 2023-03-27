import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import sentencepiece as sp
import pandas as pd
from dataclasses import dataclass

class TextGenerationDataset(Dataset):

	def __init__(self, file, block_size=512, tokenizer=None):
		super().__init__()
		''' Create a text generation dataset with the source file containing the plain text. '''
		self.block_size = block_size
		print('Initializing Dataset')
		print(f'Reading {file}...')
		with open(file, encoding='utf8') as f:
			self._raw_text = f.read()
		print('Tokenizing text...')
		tokenizer = tokenizer or (lambda x:x)
		self._vectorized_text = tokenizer(self._raw_text)
		self._vectorized_text = torch.tensor(self._vectorized_text, dtype=torch.long)

	def __getitem__(self, idx):
		x = self._vectorized_text[idx: idx+self.block_size]
		y = self._vectorized_text[idx+1: idx+self.block_size+1]
		return x, y

	def __len__(self):
		return self._vectorized_text.shape[0] - self.block_size

@dataclass
class TranslationBatch:
	x_src: torch.Tensor
	x_tgt: torch.Tensor
	x_src_mask: torch.Tensor
	x_tgt_mask: torch.Tensor
	y_tgt: torch.Tensor

	@staticmethod
	def make_batch(batch):
		# batch is a list of tuples (src, tgt)
		x_src, x_tgt, y_tgt = zip(*batch)
		# convert src & tgt to tensors
		x_src = pad_sequence(x_src, batch_first=True, padding_value=TranslationDataset.PAD_IDX)
		x_tgt = pad_sequence(x_tgt, batch_first=True, padding_value=TranslationDataset.PAD_IDX)
		y_tgt = pad_sequence(y_tgt, batch_first=True, padding_value=TranslationDataset.PAD_IDX)
		# create src_mask & tgt_mask
		x_src_tok_mask = TranslationDataset.make_pad_mask(x_src)
		x_tgt_tok_mask = TranslationDataset.make_pad_mask(x_tgt)
		return TranslationBatch(x_src, x_tgt, x_src_tok_mask, x_tgt_tok_mask, y_tgt)

class TranslationDataset(Dataset):

	UNK_IDX = 0
	BOS_IDX = 1
	EOS_IDX = 2
	PAD_IDX = 3

	def __init__(self, train_src_file: str, train_tgt_file: str, sp_model: str):
		super().__init__()
		print('Reading input files...')
		with open(train_src_file, encoding='utf8') as f:
			src_lines = list(f)
		with open(train_tgt_file, encoding='utf8') as f:
			tgt_lines = list(f)
		self.df = pd.DataFrame({ 'src' : src_lines , 'tgt' : tgt_lines })
		self.tokenizer = sp.SentencePieceProcessor(model_file=sp_model)

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