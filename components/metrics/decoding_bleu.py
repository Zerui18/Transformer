import torch
from torch import Tensor
from nltk.translate.bleu_score import sentence_bleu

from components.metrics.base_metric import BaseMetric
from components.tokenizers.base_tokenizer import BaseTokenizer


class DecodingBLEUMetric(BaseMetric):
	''' BLEU-4 score computed from autoregressively decoded token sequences.

	An epoch-frequency metric that consumes decoded token lists (produced by
	the model's produce() method) and reference token ids. Detokenizes and
	re-tokenizes both sides for normalization before computing sentence BLEU.

	Attributes:
		is_differentiable: ``bool``: False — BLEU is non-differentiable.
		higher_is_better: ``bool``: True — higher BLEU is better.
		tokenizer: ``BaseTokenizer``: used for detokenize/retokenize normalization.
		decode_key: ``str``: the produce() output key containing decoded token lists.
	'''

	is_differentiable: bool = False
	higher_is_better: bool = True

	@property
	def requires(self) -> set[str]:
		''' Keys needed from model.produce(): the decoded token lists and reference targets. '''
		return {self.decode_key, 'y_true'}

	def __init__(self,
				 tokenizer: BaseTokenizer,
				 decode_key: str = 'decoded_greedy',
				 name: str | None = None,
				 num_samples: int = 32):
		''' Initialize the decoding BLEU metric.

		Args:
			tokenizer: ``BaseTokenizer``: tokenizer for detokenize/retokenize normalization.
			decode_key: ``str``: which produce() key holds the decoded token lists.
			name: ``str | None``: display name for logging (defaults to class name).
			num_samples: ``int``: number of validation samples to decode at epoch end.
		'''
		super().__init__(name=name, frequency='epoch', num_samples=num_samples)
		self.tokenizer = tokenizer
		self.decode_key = decode_key
		self.add_state('bleu_scores', default=[], dist_reduce_fx=None)

	def _update(self, **kwargs) -> None:
		''' Accumulate per-sentence BLEU scores from decoded outputs.

		Args:
			**kwargs: must contain self.decode_key (list[list[int]]) and 'y_true' (Tensor).

		The decode_key value is a list of decoded token lists (one per sample in the batch).
		y_true is the reference target tensor (B, T).
		'''
		decoded_batch: list[list[int]] = kwargs[self.decode_key]
		y_true: Tensor = kwargs['y_true']
		B = y_true.size(0)

		for i in range(B):
			pred_tokens = decoded_batch[i]
			ref_tokens = y_true[i].cpu().tolist()
			score = self._sentence_bleu(pred_tokens, ref_tokens)
			self.bleu_scores.append(score)

	def _sentence_bleu(self, pred: list[int], reference: list[int]) -> float:
		''' Compute BLEU-4 for a single pair via detokenize-retokenize normalization.

		Args:
			pred: ``list[int]``: decoded prediction token ids.
			reference: ``list[int]``: reference token ids.
		Returns:
			``float``: BLEU-4 score in [0, 1].
		'''
		pred_text = self.tokenizer.detokenize(pred)
		ref_text = self.tokenizer.detokenize(reference)
		pred_norm = self.tokenizer.tokenize(pred_text, add_special_tokens=False)
		ref_norm = self.tokenizer.tokenize(ref_text, add_special_tokens=False)
		if not pred_norm or not ref_norm:
			return 0.0
		return sentence_bleu([ref_norm], pred_norm)

	def compute(self) -> Tensor:
		''' Compute corpus-level average BLEU-4 score.

		Returns:
			``Tensor[(), float32]``: average BLEU-4 score.
		'''
		if not self.bleu_scores:
			return torch.tensor(0.0)
		return torch.tensor(sum(self.bleu_scores) / len(self.bleu_scores))
