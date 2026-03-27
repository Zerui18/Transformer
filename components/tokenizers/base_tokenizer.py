import abc


class BaseTokenizer(abc.ABC):
	''' Abstract base for all zlab tokenizers.

	Provides public tokenize()/detokenize() methods that handle pre/post-processing
	(special tokens, padding, truncation). Subclasses implement _tokenize() and
	_detokenize() for core encoding/decoding logic only.

	Attributes:
		bos_token_id: ``int | None``: beginning-of-sequence token id.
		eos_token_id: ``int | None``: end-of-sequence token id.
		pad_token_id: ``int | None``: padding token id.
		max_length: ``int | None``: maximum sequence length (including special tokens).
	'''

	def __init__(self,
				 bos_token_id: int | None = None,
				 eos_token_id: int | None = None,
				 pad_token_id: int | None = None,
				 max_length: int | None = None):
		''' Initialize the base tokenizer.

		Args:
			bos_token_id: ``int | None``: beginning-of-sequence token id, prepended if set.
			eos_token_id: ``int | None``: end-of-sequence token id, appended if set.
			pad_token_id: ``int | None``: padding token id, used for right-padding if set.
			max_length: ``int | None``: maximum sequence length; truncates if exceeded.
		'''
		self.bos_token_id = bos_token_id
		self.eos_token_id = eos_token_id
		self.pad_token_id = pad_token_id
		self.max_length = max_length

	def tokenize(self, text: str, add_special_tokens: bool = True, pad: bool = False) -> list[int]:
		''' Tokenize text with optional special tokens, truncation, and padding.

		Args:
			text: ``str``: the input text to tokenize.
			add_special_tokens: ``bool``: whether to prepend BOS and append EOS.
			pad: ``bool``: whether to right-pad to max_length.
		Returns:
			``list[int]``: encoded token ids.
		'''
		tokens = self._tokenize(text)

		# prepend BOS, append EOS
		if add_special_tokens:
			if self.bos_token_id is not None:
				tokens = [self.bos_token_id] + tokens
			if self.eos_token_id is not None:
				tokens = tokens + [self.eos_token_id]

		# truncate to max_length
		if self.max_length is not None and len(tokens) > self.max_length:
			tokens = tokens[:self.max_length]

		# right-pad to max_length
		if pad and self.max_length is not None and self.pad_token_id is not None:
			tokens = tokens + [self.pad_token_id] * (self.max_length - len(tokens))

		return tokens

	def detokenize(self, tokens: list[int], strip_special_tokens: bool = True) -> str:
		''' Decode token ids back to text, optionally stripping special tokens.

		Args:
			tokens: ``list[int]``: token ids to decode.
			strip_special_tokens: ``bool``: whether to remove BOS/EOS/PAD before decoding.
		Returns:
			``str``: decoded text.
		'''
		if strip_special_tokens:
			special = {t for t in (self.bos_token_id, self.eos_token_id, self.pad_token_id) if t is not None}
			tokens = [t for t in tokens if t not in special]

		return self._detokenize(tokens)

	@abc.abstractmethod
	def _tokenize(self, text: str) -> list[int]:
		''' Core tokenization logic. Subclasses must implement this.

		Args:
			text: ``str``: raw input text.
		Returns:
			``list[int]``: encoded token ids (without special tokens).
		'''
		...

	@abc.abstractmethod
	def _detokenize(self, tokens: list[int]) -> str:
		''' Core detokenization logic. Subclasses must implement this.

		Args:
			tokens: ``list[int]``: token ids (already stripped of special tokens).
		Returns:
			``str``: decoded text.
		'''
		...
