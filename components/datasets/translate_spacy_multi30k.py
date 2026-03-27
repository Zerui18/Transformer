from typing import Callable, Iterable

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k

from components.datasets.base_dataset import BaseDataset


# Define special symbols and indices
UNK_IDX, BOS_IDX, EOS_IDX, PAD_IDX = 0, 1, 2, 3


def sequential_transforms(*transforms: Callable) -> Callable:
	''' Chain multiple transforms into a single callable that applies them sequentially.

	Args:
		*transforms: ``Callable``: variable number of callables to be applied in order.
	Returns:
		``Callable``: composed transform function.
	'''
	def func(txt_input: str) -> Tensor:
		for transform in transforms:
			txt_input = transform(txt_input)
		return txt_input
	return func


def tensor_transform(token_ids: list[int]) -> Tensor:
	''' Wrap token ids with BOS/EOS and convert to a tensor.

	Args:
		token_ids: ``list[int]``: list of integer token indices.
	Returns:
		``Tensor[(T,), int64]``: tensor with BOS prepended and EOS appended.
	'''
	return torch.cat((
		torch.tensor([BOS_IDX]),
		torch.tensor(token_ids),
		torch.tensor([EOS_IDX]),
	))


def yield_tokens(data_iter: Iterable, language: str) -> Iterable[list[str]]:
	''' Yield lists of tokens from a data iterator for a given language.

	Args:
		data_iter: ``Iterable``: iterable of (source, target) text pairs.
		language: ``str``: language key ('de' or 'en').
	Yields:
		``list[str]``: tokenized text for the specified language.
	'''
	language_index = {'de': 0, 'en': 1}
	for data_sample in data_iter:
		yield TranslationDatasetSpacyMulti30K.token_transform[language](data_sample[language_index[language]])


class TranslationDatasetSpacyMulti30K(BaseDataset):
	''' Translation dataset using spaCy tokenizers on the Multi30k dataset from torchtext.

	Attributes:
		src_lang: ``str``: source language code.
		tgt_lang: ``str``: target language code.
		split: ``str``: dataset split ('train' or 'valid').
		dataset: ``list[tuple[str, str]]``: list of (source, target) sentence pairs.

	Currently only de -> en is supported as the vocabulary building and text
	transforms are hardcoded for this language pair. Class-level resources
	(tokenizers, vocabularies, text transforms) are initialized once on first
	instantiation and shared across all instances.
	'''

	_INIT_RESOURCES_DONE: bool = False
	token_transform: dict[str, Callable] | None = None
	vocab_transform: dict[str, Callable] | None = None
	text_transform: dict[str, Callable] | None = None

	def __init__(
		self,
		src_language: str,
		tgt_language: str,
		split: str,
	) -> None:
		''' Initialise the Multi30k translation dataset for the given language pair and split.

		Args:
			src_language: ``str``: source language code (e.g. 'de').
			tgt_language: ``str``: target language code (e.g. 'en').
			split: ``str``: dataset split, one of 'train' or 'valid'.
		'''
		TranslationDatasetSpacyMulti30K._init_resources()
		super().__init__()
		self.src_lang = src_language
		self.tgt_lang = tgt_language
		self.split = split
		self.dataset: list[tuple[str, str]] = list(
			Multi30k(split=self.split, language_pair=(self.src_lang, self.tgt_lang))
		)

	@staticmethod
	def _init_resources() -> None:
		''' Build shared tokenizers, vocabularies, and text transforms on first call.

		This is a one-time initialization that patches Multi30k URLs, creates
		spaCy tokenizers, builds vocabularies from the training split, and
		composes the full text transform pipeline. Subsequent calls are no-ops.
		'''
		if TranslationDatasetSpacyMulti30K._INIT_RESOURCES_DONE:
			return

		# Patch URLs for broken links
		multi30k.URL["train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
		multi30k.URL["valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"

		token_transform: dict[str, Callable] = {}
		vocab_transform: dict[str, Callable] = {}

		SRC_LANG = 'de'
		TGT_LANG = 'en'

		token_transform[SRC_LANG] = get_tokenizer('spacy', language='de_core_news_sm')
		token_transform[TGT_LANG] = get_tokenizer('spacy', language='en_core_web_sm')

		TranslationDatasetSpacyMulti30K.token_transform = token_transform

		special_symbols = ['<unk>', '<bos>', '<eos>', '<pad>']

		for ln in [SRC_LANG, TGT_LANG]:
			train_iter = Multi30k(split='train', language_pair=(SRC_LANG, TGT_LANG))
			vocab_transform[ln] = build_vocab_from_iterator(
				yield_tokens(train_iter, ln),
				min_freq=1,
				specials=special_symbols,
				special_first=True,
			)

		# Set UNK_IDX as the default index for out-of-vocabulary tokens
		for ln in [SRC_LANG, TGT_LANG]:
			vocab_transform[ln].set_default_index(UNK_IDX)

		TranslationDatasetSpacyMulti30K.vocab_transform = vocab_transform

		# Compose full text transform: tokenize -> numericalize -> add BOS/EOS
		text_transform: dict[str, Callable] = {}
		for ln in [SRC_LANG, TGT_LANG]:
			text_transform[ln] = sequential_transforms(
				token_transform[ln],
				vocab_transform[ln],
				tensor_transform,
			)

		TranslationDatasetSpacyMulti30K.text_transform = text_transform
		TranslationDatasetSpacyMulti30K._INIT_RESOURCES_DONE = True

		print(f"Vocab size for {SRC_LANG}: {len(vocab_transform[SRC_LANG])}")
		print(f"Vocab size for {TGT_LANG}: {len(vocab_transform[TGT_LANG])}")

	def __len__(self) -> int:
		''' Return the number of samples in the current split.

		Returns:
			``int``: number of sentence pairs (29000 for train, 1014 for valid).
		'''
		if self.split == 'train':
			return 29000
		elif self.split == 'valid':
			return 1014
		return 0

	def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
		''' Return transformed source and target tensors for a given index.

		Args:
			idx: ``int``: sample index.
		Returns:
			``Tensor[(Ts,), int64]``: source tokens with BOS and EOS.
			``Tensor[(Tt,), int64]``: target tokens with BOS and EOS.
		'''
		src, dst = self.dataset[idx]
		src = TranslationDatasetSpacyMulti30K.text_transform['de'](src.rstrip("\n"))
		dst = TranslationDatasetSpacyMulti30K.text_transform['en'](dst.rstrip("\n"))
		return src, dst

	@staticmethod
	def get_collate_function() -> Callable | None:
		''' Return a collate function that pads sequences and splits target into input/label.

		Returns:
			``Callable``: collate function producing dict[str, Tensor].
		'''
		def collate_fn(batch: list[tuple[Tensor, Tensor]]) -> dict[str, Tensor]:
			''' Collate a list of (src, tgt) pairs into a padded batch dict.

			Args:
				batch: ``list[tuple[Tensor, Tensor]]``: list of (source, target) tensor pairs.
			Returns:
				``dict[str, Tensor]``: with keys:
					'x_src' [int64, (B, Ts)], 'x_tgt' [int64, (B, Tt-1)],
					'x_src_mask' [bool, (B, Ts)], 'x_tgt_mask' [bool, (B, Tt-1)],
					'y_tgt' [int64, (B, Tt-1)].
			'''
			src_batch, tgt_batch = zip(*batch)
			src_batch = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
			tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)
			return {
				'x_src': src_batch,
				'x_tgt': tgt_batch[:, :-1].clone(),
				'x_src_mask': src_batch != PAD_IDX,
				'x_tgt_mask': tgt_batch[:, :-1] != PAD_IDX,
				'y_tgt': tgt_batch[:, 1:].clone(),
			}
		return collate_fn
