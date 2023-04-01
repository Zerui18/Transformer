import torch
from torch.nn.utils.rnn import pad_sequence
import sentencepiece as sp
from dataclasses import dataclass
from .base import BaseDataset

@dataclass
class TextGenDatasetConfig:
	sp_model_file: str
	text_file: str
	block_size: int

class TextGenDataset(BaseDataset):

	PAD_IDX = 0

	def __init__(self, config: TextGenDatasetConfig):
		super().__init__()
		''' Create a text generation dataset with the source file containing the plain text. '''
		self.config = config
		print('Initializing Dataset')
		print(f'Reading {config.text_file}...')
		with open(config.text_file, encoding='utf8') as f:
			self._raw_text = f.read()
		print('Tokenizing text...')
		self.tokenizer = sp.SentencePieceProcessor(model_file=config.sp_model_file)
		self._tokenized_text = self.tokenizer.encode(self._raw_text)

	def __getitem__(self, idx):
		x = self._tokenized_text[idx: idx+self.config.block_size]
		y = self._tokenized_text[idx+1: idx+self.config.block_size+1]
		return x, y

	def __len__(self):
		return len(self._tokenized_text) - self.config.block_size
	
	@staticmethod
	def get_collate_function() -> callable or None:
		def collate_function(batch):
			x, y = zip(*batch)
			x = pad_sequence(x, batch_first=True, padding_value=TextGenDataset.PAD_IDX)
			y = pad_sequence(y, batch_first=True, padding_value=TextGenDataset.PAD_IDX)
			return x, y
		return collate_function