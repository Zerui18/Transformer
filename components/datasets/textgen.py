from typing import Callable

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from sentencepiece import SentencePieceProcessor

from components.datasets.base_dataset import BaseDataset


class TextGenDataset(BaseDataset):
	''' Text generation dataset that loads plain text, tokenizes it, and yields (input, target) blocks.

	Attributes:
		PAD_IDX: ``int``: padding token index.
		tokenizer: ``SentencePieceProcessor``: the sentencepiece tokenizer used to encode the text.
		block_size: ``int``: the number of tokens per sample.
	'''

	PAD_IDX: int = 0

	def __init__(
		self,
		sp_model_file: str,
		text_file: str,
		block_size: int,
	) -> None:
		''' Initialise the text generation dataset from a plain text file and a sentencepiece model.

		Args:
			sp_model_file: ``str``: path to the sentencepiece model file.
			text_file: ``str``: path to the plain text file to be tokenized.
			block_size: ``int``: the number of tokens per sample (context window size).
		'''
		super().__init__()
		self.block_size = block_size

		print('Initializing Dataset')
		print(f'Reading {text_file}...')
		with open(text_file, encoding='utf8') as f:
			raw_text = f.read()

		print('Tokenizing text...')
		self.tokenizer = SentencePieceProcessor(model_file=sp_model_file)
		self._tokenized_text: list[int] = self.tokenizer.encode(raw_text)

	def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
		''' Return the input and target token tensors for a given index.

		Args:
			idx: ``int``: sample index.
		Returns:
			``Tensor[(T,), int64]``: input token ids of length block_size.
			``Tensor[(T,), int64]``: target token ids shifted by one position.
		'''
		x = torch.tensor(self._tokenized_text[idx: idx + self.block_size], dtype=torch.long)
		y = torch.tensor(self._tokenized_text[idx + 1: idx + self.block_size + 1], dtype=torch.long)
		return x, y

	def __len__(self) -> int:
		''' Return the number of samples in the dataset.

		Returns:
			``int``: total number of valid (input, target) pairs.
		'''
		return len(self._tokenized_text) - self.block_size

	@staticmethod
	def get_collate_function() -> Callable | None:
		''' Return a collate function that pads variable-length sequences and returns a dict batch.

		Returns:
			``Callable``: collate function producing dict[str, Tensor].
		'''
		def collate_function(batch: list[tuple[Tensor, Tensor]]) -> dict[str, Tensor]:
			''' Collate a list of (x, y) pairs into a padded batch dict.

			Args:
				batch: ``list[tuple[Tensor, Tensor]]``: list of (input, target) tensor pairs.
			Returns:
				``dict[str, Tensor]``: with keys 'x' [int64, (B, T)] and 'y' [int64, (B, T)].
			'''
			x, y = zip(*batch)
			x = pad_sequence(x, batch_first=True, padding_value=TextGenDataset.PAD_IDX)
			y = pad_sequence(y, batch_first=True, padding_value=TextGenDataset.PAD_IDX)
			return {'x': x, 'y': y}
		return collate_function
