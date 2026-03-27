import math
from pathlib import Path
from typing import Callable

import torch
import numpy as np
import pandas as pd
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from sentencepiece import SentencePieceProcessor

from components.datasets.base_dataset import BaseDataset


class ATISDataset(BaseDataset):
	''' ATIS speech recognition dataset that loads mel spectrograms and transcripts for Whisper-style models.

	Properties:
		1. UNK_IDX: int  unknown token index.
		2. BOS_IDX: int  beginning-of-sequence token index.
		3. EOS_IDX: int  end-of-sequence token index.
		4. PAD_IDX: int  padding token index.
		5. max_seq_len: int  maximum decoder sequence length (tokens are truncated to this before special tokens).
		6. df: pd.DataFrame  dataframe with columns 'file_id', 'transcript', and 'mel'.
		7. tokenizer: SentencePieceProcessor  tokenizer for transcript text.
	'''

	UNK_IDX: int = 0
	BOS_IDX: int = 1
	EOS_IDX: int = 2
	PAD_IDX: int = 3

	def __init__(
		self,
		sp_model: str,
		mel_dir: str,
		transcripts_file: str,
		dec_max_len: int,
		first_n_lines: int | None = None,
	) -> None:
		''' Initialise the ATIS dataset from mel spectrogram files, a transcripts file, and a SentencePiece model.

		Args:
			1. sp_model: str  path to the SentencePiece model file for transcript tokenization.
			2. mel_dir: str  directory containing .npy mel spectrogram files named by file id.
			3. transcripts_file: str  path to the transcripts file (each line: 8-char file_id followed by transcript).
			4. dec_max_len: int  maximum decoder sequence length for transcript tokens.
			5. first_n_lines: int | None  if set, only load the first N lines from the transcripts file.
		'''
		super().__init__()
		self.max_seq_len = dec_max_len
		print('DS INIT:', sp_model, mel_dir, transcripts_file, dec_max_len, first_n_lines)

		# read transcripts
		with open(transcripts_file, encoding='utf8') as f:
			all_lines = f.readlines()
		lines = [line for line in all_lines if len(line.strip()) > 0]
		if first_n_lines is not None:
			lines = lines[:first_n_lines]
		file_ids, transcripts = zip(*[(line[:8], line[9:]) for line in lines])

		df = pd.DataFrame({'file_id': file_ids, 'transcript': transcripts})

		# read mels
		mel_path = Path(mel_dir)
		mel_paths = [mel_path / f'{file_id}.npy' for file_id in file_ids]
		mels = [np.load(p) if p.exists() else None for p in mel_paths]

		# add mels to df and filter out missing ones
		df['mel'] = mels
		df = df[df.mel.notnull()]

		# init tokenizer
		self.tokenizer = SentencePieceProcessor(model_file=sp_model)

		print('DS HEAD:')
		print(df.head())
		self.df = df
		print('DS INIT COMPLETE')

	def __len__(self) -> int:
		''' Return the number of samples with valid mel spectrograms.

		Returns:
			length: int  number of samples.
		'''
		return len(self.df)

	def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
		''' Return mel spectrogram source, decoder input, and decoder label tensors for a given index.

		Args:
			1. idx: int  sample index.
		Returns:
			x_src: Tensor [float32, (Ts, C)]  transposed mel spectrogram (time steps x mel channels).
			x_tgt: Tensor [int64, (Tt,)]  decoder input tokens with BOS prefix.
			y_tgt: Tensor [int64, (Tt,)]  decoder label tokens with EOS suffix.

		BPE dropout (sampling) is enabled during tokenization for regularization.
		'''
		row = self.df.iloc[idx]
		mel, transcript = row.mel, row.transcript

		# enable bpe dropout regularization
		transcript = self.tokenizer.encode(transcript, enable_sampling=True, alpha=0.1, nbest_size=-1)[:self.max_seq_len]

		# create src & tgt tensors
		x_src = torch.tensor(mel.T, dtype=torch.float)
		x_tgt = torch.tensor([ATISDataset.BOS_IDX] + transcript, dtype=torch.long)
		y_tgt = torch.tensor(transcript + [ATISDataset.EOS_IDX], dtype=torch.long)
		return x_src, x_tgt, y_tgt

	@staticmethod
	def make_pad_mask(x: Tensor) -> Tensor:
		''' Create a boolean mask that is True for non-padding positions.

		Args:
			1. x: Tensor [int64, (B, T)]  token id tensor.
		Returns:
			mask: Tensor [bool, (B, T)]  True where token is not PAD.
		'''
		return (x != ATISDataset.PAD_IDX)

	@staticmethod
	def make_pad_mask_with_lengths(max_len: int, lengths: Tensor) -> Tensor:
		''' Create a boolean mask from sequence lengths.

		Args:
			1. max_len: int  maximum sequence length for the mask width.
			2. lengths: Tensor [int64, (B,)]  actual lengths of each sequence in the batch.
		Returns:
			mask: Tensor [bool, (B, max_len)]  True for positions within each sequence length.
		'''
		return torch.arange(max_len)[None, :] < lengths[:, None]

	@staticmethod
	def get_collate_function() -> Callable | None:
		''' Return a collate function that pads mel spectrograms and token sequences.

		Returns:
			collate_fn: Callable  collate function producing dict[str, Tensor].
		'''
		def collate_function(batch: list[tuple[Tensor, Tensor, Tensor]]) -> dict[str, Tensor]:
			''' Collate a list of (x_src, x_tgt, y_tgt) tuples into a padded batch dict.

			Args:
				1. batch: list[tuple[Tensor, Tensor, Tensor]]  list of sample tuples.
			Returns:
				result: dict[str, Tensor]  with keys:
					'x_src' [float32, (B, Ts, C)], 'x_tgt' [int64, (B, Tt)],
					'x_src_mask' [bool, (B, Ts//2)], 'x_tgt_mask' [bool, (B, Tt)],
					'y_tgt' [int64, (B, Tt)].

			The source mask accounts for downsampling by a factor of 2 in the encoder.
			'''
			x_src, x_tgt, y_tgt = zip(*batch)

			# track source lengths for mask computation
			lengths = torch.tensor([len(x) for x in x_src], dtype=torch.long)

			# pad sequences
			x_src = pad_sequence(x_src, batch_first=True, padding_value=0.0)
			x_tgt = pad_sequence(x_tgt, batch_first=True, padding_value=ATISDataset.PAD_IDX)
			y_tgt = pad_sequence(y_tgt, batch_first=True, padding_value=ATISDataset.PAD_IDX)

			# create masks (source mask accounts for 2x downsampling in encoder)
			x_src_mask = ATISDataset.make_pad_mask_with_lengths(
				math.ceil(x_src.shape[1] / 2),
				torch.ceil(lengths // 2),
			)
			x_tgt_mask = ATISDataset.make_pad_mask(x_tgt)

			return {
				'x_src': x_src,
				'x_tgt': x_tgt,
				'x_src_mask': x_src_mask,
				'x_tgt_mask': x_tgt_mask,
				'y_tgt': y_tgt,
			}
		return collate_function
