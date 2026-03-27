from typing import Callable

import torch
import pandas as pd
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from sentencepiece import SentencePieceProcessor

from components.datasets.base_dataset import BaseDataset


class TranslationDataset(BaseDataset):
	''' Translation dataset that loads parallel source/target text files and tokenizes with SentencePiece.

	Attributes:
		UNK_IDX: ``int``: unknown token index.
		BOS_IDX: ``int``: beginning-of-sequence token index.
		EOS_IDX: ``int``: end-of-sequence token index.
		PAD_IDX: ``int``: padding token index.
		max_seq_len: ``int``: maximum sequence length (tokens are truncated to this length before adding special tokens).
		df: ``pd.DataFrame``: dataframe holding source and target text lines.
		src_tokenizer: ``SentencePieceProcessor``: tokenizer for source language.
		tgt_tokenizer: ``SentencePieceProcessor``: tokenizer for target language.
	'''

	UNK_IDX: int = 0
	BOS_IDX: int = 1
	EOS_IDX: int = 2
	PAD_IDX: int = 3

	def __init__(
		self,
		src_sp_model_file: str,
		tgt_sp_model_file: str,
		src_file: str,
		tgt_file: str,
		max_seq_len: int,
		first_n_lines: int | None = None,
	) -> None:
		''' Initialise the translation dataset from parallel text files and SentencePiece models.

		Args:
			src_sp_model_file: ``str``: path to the source language SentencePiece model file.
			tgt_sp_model_file: ``str``: path to the target language SentencePiece model file.
			src_file: ``str``: path to the source language text file (one sentence per line).
			tgt_file: ``str``: path to the target language text file (one sentence per line).
			max_seq_len: ``int``: maximum number of content tokens per sequence (before BOS/EOS).
			first_n_lines: ``int | None``: if set, only load the first N lines from each file.
		'''
		super().__init__()
		self.max_seq_len = max_seq_len

		print('Reading input files...')
		with open(src_file, encoding='utf8') as f:
			if first_n_lines is None:
				src_lines = [l.strip() for l in f]
			else:
				src_lines = [next(f).strip() for _ in range(first_n_lines)]
		with open(tgt_file, encoding='utf8') as f:
			if first_n_lines is None:
				tgt_lines = [l.strip() for l in f]
			else:
				tgt_lines = [next(f).strip() for _ in range(first_n_lines)]

		self.df = pd.DataFrame({'src': src_lines, 'tgt': tgt_lines})
		print(self.df.head(10))

		print('Loading sentencepiece models...')
		self.src_tokenizer = SentencePieceProcessor(model_file=src_sp_model_file)
		self.tgt_tokenizer = SentencePieceProcessor(model_file=tgt_sp_model_file)
		print('Done.')

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
			``Tensor[(Ts,), int64]``: source tokens with BOS and EOS.
			``Tensor[(Tt,), int64]``: target input tokens with BOS prefix (teacher forcing).
			``Tensor[(Tt,), int64]``: target label tokens with EOS suffix.

		BPE dropout (sampling) is enabled during tokenization for regularization.
		'''
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
	def make_pad_mask(x: Tensor) -> Tensor:
		''' Create a boolean mask that is True for non-padding positions.

		Args:
			x: ``Tensor[(B, T), int64]``: token id tensor.
		Returns:
			``Tensor[(B, T), bool]``: True where token is not PAD.
		'''
		return (x != TranslationDataset.PAD_IDX)

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
			x_src = pad_sequence(x_src, batch_first=True, padding_value=TranslationDataset.PAD_IDX)
			x_tgt = pad_sequence(x_tgt, batch_first=True, padding_value=TranslationDataset.PAD_IDX)
			y_tgt = pad_sequence(y_tgt, batch_first=True, padding_value=TranslationDataset.PAD_IDX)
			# create masks
			x_src_mask = TranslationDataset.make_pad_mask(x_src)
			x_tgt_mask = TranslationDataset.make_pad_mask(x_tgt)
			return {
				'x_src': x_src,
				'x_tgt': x_tgt,
				'x_src_mask': x_src_mask,
				'x_tgt_mask': x_tgt_mask,
				'y_tgt': y_tgt,
			}
		return collate_function
