from pickle import load
from typing import Callable

import torch
import pandas as pd
import spacy
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from components.datasets.base_dataset import BaseDataset


class TranslationDatasetSpacy(BaseDataset):
	''' Translation dataset using spaCy tokenizers and pre-built pickle vocabularies.

	Attributes:
		UNK_IDX: ``int``: unknown token index.
		BOS_IDX: ``int``: beginning-of-sequence token index.
		EOS_IDX: ``int``: end-of-sequence token index.
		PAD_IDX: ``int``: padding token index.
		max_seq_len: ``int``: maximum sequence length for character truncation before tokenization.
		df: ``pd.DataFrame``: dataframe holding source and target text lines.
		src_tokenizer: ``spacy.language.Language``: spaCy tokenizer for source language.
		tgt_tokenizer: ``spacy.language.Language``: spaCy tokenizer for target language.
		src_vocab: ``dict``: source vocabulary mapping token strings to indices.
		tgt_vocab: ``dict``: target vocabulary mapping token strings to indices.
	'''

	UNK_IDX: int = 0
	BOS_IDX: int = 1
	EOS_IDX: int = 2
	PAD_IDX: int = 3

	def __init__(
		self,
		src_model: str,
		tgt_model: str,
		src_vocab_file: str,
		tgt_vocab_file: str,
		src_file: str,
		tgt_file: str,
		max_seq_len: int,
		first_n_lines: int | None = None,
	) -> None:
		''' Initialise the translation dataset from parallel text files, spaCy models, and pickled vocabs.

		Args:
			src_model: ``str``: name of the spaCy model for the source language.
			tgt_model: ``str``: name of the spaCy model for the target language.
			src_vocab_file: ``str``: path to the pickled source vocabulary file.
			tgt_vocab_file: ``str``: path to the pickled target vocabulary file.
			src_file: ``str``: path to the source language text file (one sentence per line).
			tgt_file: ``str``: path to the target language text file (one sentence per line).
			max_seq_len: ``int``: maximum character length for truncation before tokenization.
			first_n_lines: ``int | None``: if set, only load the first N lines from each file.
		'''
		super().__init__()
		self.max_seq_len = max_seq_len

		print('Reading input files...')
		with open(src_file, encoding='utf8') as f:
			if first_n_lines is None:
				src_lines = f.readlines()
			else:
				src_lines = [next(f) for _ in range(first_n_lines)]
		with open(tgt_file, encoding='utf8') as f:
			if first_n_lines is None:
				tgt_lines = f.readlines()
			else:
				tgt_lines = [next(f) for _ in range(first_n_lines)]

		self.df = pd.DataFrame({'src': src_lines, 'tgt': tgt_lines})

		print('Initializing tokenizers...')
		self.src_tokenizer = spacy.load(src_model)
		self.tgt_tokenizer = spacy.load(tgt_model)

		print('Initializing vocabularies...')
		with open(src_vocab_file, 'rb') as f:
			self.src_vocab: dict = load(f)
		with open(tgt_vocab_file, 'rb') as f:
			self.tgt_vocab: dict = load(f)

		print('Dataset initialized.')

	def __len__(self) -> int:
		''' Return the number of parallel sentence pairs.

		Returns:
			``int``: number of samples.
		'''
		return len(self.df)

	def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
		''' Return source input, target input, and target label tensors for a given index.

		Args:
			idx: ``int``: sample index.
		Returns:
			``Tensor[(Ts,), int64]``: source token indices.
			``Tensor[(Tt,), int64]``: target input tokens with BOS prefix.
			``Tensor[(Tt,), int64]``: target label tokens with EOS suffix.
		'''
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
	def make_pad_mask(x: Tensor) -> Tensor:
		''' Create a boolean mask that is True for non-padding positions.

		Args:
			x: ``Tensor[(B, T), int64]``: token id tensor.
		Returns:
			``Tensor[(B, T), bool]``: True where token is not PAD.
		'''
		return (x != TranslationDatasetSpacy.PAD_IDX)

	@staticmethod
	def get_collate_function() -> Callable | None:
		''' Return a collate function that pads sequences and produces mask tensors.

		Returns:
			``Callable``: collate function producing dict[str, Tensor].
		'''
		def collate_function(batch: list[tuple[Tensor, Tensor, Tensor]]) -> dict[str, Tensor]:
			''' Collate a list of (x_src, x_tgt, y_tgt) tuples into a padded batch dict.

			Args:
				batch: ``list[tuple[Tensor, Tensor, Tensor]]``: list of sample tuples.
			Returns:
				``dict[str, Tensor]``: with keys:
					'x_src' [int64, (B, Ts)], 'x_tgt' [int64, (B, Tt)],
					'x_src_mask' [bool, (B, Ts)], 'x_tgt_mask' [bool, (B, Tt)],
					'y_tgt' [int64, (B, Tt)].
			'''
			x_src, x_tgt, y_tgt = zip(*batch)
			# pad sequences
			x_src = pad_sequence(x_src, batch_first=True, padding_value=TranslationDatasetSpacy.PAD_IDX)
			x_tgt = pad_sequence(x_tgt, batch_first=True, padding_value=TranslationDatasetSpacy.PAD_IDX)
			y_tgt = pad_sequence(y_tgt, batch_first=True, padding_value=TranslationDatasetSpacy.PAD_IDX)
			# create masks
			x_src_mask = TranslationDatasetSpacy.make_pad_mask(x_src)
			x_tgt_mask = TranslationDatasetSpacy.make_pad_mask(x_tgt)
			return {
				'x_src': x_src,
				'x_tgt': x_tgt,
				'x_src_mask': x_src_mask,
				'x_tgt_mask': x_tgt_mask,
				'y_tgt': y_tgt,
			}
		return collate_function
