from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
import sentencepiece as sp
import torch.nn.functional as F
import pandas as pd

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
	src: torch.Tensor
	tgt: torch.Tensor
	src_mask: torch.Tensor
	tgt_mask: torch.Tensor

	@staticmethod
	def make_batch(batch):
		# batch is a list of tuples (src, tgt)
		x, y_tgt = zip(*batch)
		x_src, x_tgt = zip(*x)
		# convert src & tgt to tensors
		x_src = torch.stack(x_src)
		x_tgt = torch.stack(x_tgt)
		y_tgt = torch.stack(y_tgt)
		# create src_mask & tgt_mask
		x_src_mask = TranslationDataset.make_pad_mask(x_src)
		x_tgt_mask = TranslationDataset.make_pad_mask(x_tgt)
		return TranslationBatch(x_src, x_tgt, x_src_mask, x_tgt_mask)

class TranslationDataset(Dataset):

	UNK_IDX = 0
	BOS_IDX = 1
	EOS_IDX = 2
	PAD_IDX = 3

	def __init__(self, train_src_file: str, train_tgt_file: str, sp_model: str, block_size: int):
		super().__init__()
		print('Reading input files...')
		with open(train_src_file, encoding='utf8') as f:
			src_lines = list(f)
		with open(train_tgt_file, encoding='utf8') as f:
			tgt_lines = list(f)
		self.df = pd.DataFrame({ 'src' : src_lines , 'tgt' : tgt_lines })
		self.tokenizer = sp.SentencePieceProcessor(model_file=sp_model)
		self.block_size = block_size

	def __len__(self):
		return len(self.df)
	
	def __getitem__(self, idx):
		row = self.df.iloc[idx]
		src, tgt = row.src, row.tgt
		# enable bpe dropout regularization
		x_src = self.tokenizer.encode(src, enable_sampling=True, alpha=0.1, nbest_size=-1)
		x_tgt = self.tokenizer.encode(tgt, enable_sampling=True, alpha=0.1, nbest_size=-1)
		# convert src, tgt to tensors of block_size
		x_src = torch.tensor(x_src, dtype=torch.long)
		x_tgt = torch.tensor(x_tgt, dtype=torch.long)
		# apply clipping & padding to convert to block_size
		x_src = self.pad_to_length(x_src, self.block_size)
		x_tgt = self.pad_to_length(x_tgt, self.block_size - 1)
		x_tgt = torch.concat((torch.tensor([TranslationDataset.BOS_IDX]), x_tgt))
		y_tgt = self.pad_to_length(x_tgt, self.block_size - 1)
		y_tgt = torch.concat((y_tgt, torch.tensor([TranslationDataset.EOS_IDX])))
		return (x_src, x_tgt), y_tgt
	
	@staticmethod
	def pad_to_length(x: torch.Tensor, length: int):
		''' Clip or pad a tensor to a given length. '''
		if x.shape[0] > length:
			return x[:length]
		return F.pad(x, pad=(0, length-x.shape[0]), mode='constant', value=TranslationDataset.PAD_IDX)

	@staticmethod
	def make_pad_mask(x):
		return (x != TranslationDataset.PAD_IDX).unsqueeze(1) # (B, 1, T)