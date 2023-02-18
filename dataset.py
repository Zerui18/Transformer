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
		with open(file) as f:
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
		src, tgt = zip(*batch)
		# convert src & tgt to tensors
		src = torch.stack(src)
		tgt = torch.stack(tgt)
		# create src_mask & tgt_mask
		src_mask = TranslationDataset.make_pad_mask(src)
		tgt_mask = TranslationDataset.make_pad_mask(tgt)
		return TranslationBatch(src, tgt, src_mask, tgt_mask)

class TranslationDataset(Dataset):

	UNK_IDX = 0
	BOS_IDX = 1
	EOS_IDX = 2
	PAD_IDX = 3

	def __init__(self, train_src_file: str, train_tgt_file: str, sp_model: str, block_size: int):
		super().__init__()
		print('Reading input files...')
		with open(train_src_file) as f:
			src_lines = list(f)
		with open(train_tgt_file) as f:
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
		src = [self.BOS_IDX] + self.tokenizer.encode(src, enable_sampling=True, alpha=0.1, nbest_size=-1) + [self.EOS_IDX]
		tgt = [self.BOS_IDX] + self.tokenizer.encode(tgt, enable_sampling=True, alpha=0.1, nbest_size=-1) + [self.EOS_IDX]
		# convert src, tgt to tensors of block_size
		src = torch.tensor(src, dtype=torch.long)
		tgt = torch.tensor(tgt, dtype=torch.long)
		# apply clipping & padding to convert to block_size
		src_len, tgt_len = src.shape[0], tgt.shape[0]
		if src_len > self.block_size:
			src = src[:self.block_size]
		elif src_len < self.block_size:
			src = F.pad(src, pad=(0, self.block_size-src_len), mode='constant', value=self.PAD_IDX)
		if tgt_len > self.block_size:
			tgt = tgt[:self.block_size]
		elif tgt_len < self.block_size:
			tgt = F.pad(tgt, pad=(0, self.block_size-tgt_len), mode='constant', value=self.PAD_IDX)
		return src, tgt
	
	@staticmethod
	def make_pad_mask(x):
		return (x != TranslationDataset.PAD_IDX).unsqueeze(1) # (B, 1, T)