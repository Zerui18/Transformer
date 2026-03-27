import numpy as np
import torch
from torch import Tensor
from nltk.translate.bleu_score import sentence_bleu

from components.metrics.base_metric import BaseMetric
from components.tokenizers.base_tokenizer import BaseTokenizer

UNK_IDX = 0
PAD_IDX = 3


class BLEUMetric(BaseMetric):
	''' Corpus-level BLEU-4 score accumulated over batches.

	Detokenizes predictions and references, re-tokenizes to normalize, then
	computes sentence-level BLEU-4 via NLTK and averages across the corpus.

	Properties:
		1. is_differentiable: bool  False — BLEU is non-differentiable.
		2. higher_is_better: bool  True — higher BLEU indicates better translation.
		3. tokenizer: BaseTokenizer  tokenizer used for detokenize/retokenize normalization.
	'''

	is_differentiable: bool = False
	higher_is_better: bool = True

	@property
	def requires(self) -> set[str]:
		''' Keys needed from model.produce(): y_pred logits and y_true token ids. '''
		return {'y_pred', 'y_true'}

	def __init__(self, tokenizer: BaseTokenizer, subsample_rate: float | None = None):
		''' Initialize the BLEU metric.

		Args:
			1. tokenizer: BaseTokenizer  tokenizer for detokenize/retokenize normalization.
			2. subsample_rate: float | None  fraction of updates to keep. None = all.
		'''
		super().__init__(subsample_rate=subsample_rate)
		self.tokenizer = tokenizer
		self.add_state('bleu_scores', default=[], dist_reduce_fx=None)

	def _update(self, y_pred: Tensor, y_true: Tensor) -> None:
		''' Accumulate per-sentence BLEU scores from a batch.

		Args:
			1. y_pred: Tensor  [float32, (B, T, V)] predicted logits.
			2. y_true: Tensor  [int64, (B, T)] ground truth token ids.
		'''
		# argmax to get predicted token ids
		pred_ids = y_pred.argmax(dim=-1)  # (B, T)
		B = pred_ids.size(0)
		for i in range(B):
			pred = pred_ids[i].cpu().numpy()
			ref = y_true[i].cpu().numpy()
			score = self._sentence_bleu(pred, ref)
			self.bleu_scores.append(score)

	def _sentence_bleu(self, pred: np.ndarray, reference: np.ndarray) -> float:
		''' Compute BLEU-4 for a single prediction/reference pair.

		Args:
			1. pred: np.ndarray  [int64, (Tp,)] predicted token ids.
			2. reference: np.ndarray  [int64, (Tr,)] reference token ids.
		Returns:
			score: float  BLEU-4 score in [0, 1].
		'''
		# clean: remove UNK and PAD tokens
		pred = pred[(pred != UNK_IDX) & (pred != PAD_IDX)]
		reference = reference[(reference != UNK_IDX) & (reference != PAD_IDX)]
		# detokenize then retokenize to normalize
		pred_text = self.tokenizer.detokenize([int(t) for t in pred])
		ref_text = self.tokenizer.detokenize([int(t) for t in reference])
		pred_tokens = self.tokenizer.tokenize(pred_text, add_special_tokens=False)
		ref_tokens = self.tokenizer.tokenize(ref_text, add_special_tokens=False)
		return sentence_bleu([ref_tokens], pred_tokens)

	def compute(self) -> Tensor:
		''' Compute corpus-level average BLEU-4 score.

		Returns:
			bleu: Tensor  [float32, ()] average BLEU-4 score.
		'''
		if not self.bleu_scores:
			return torch.tensor(0.0)
		return torch.tensor(sum(self.bleu_scores) / len(self.bleu_scores))
