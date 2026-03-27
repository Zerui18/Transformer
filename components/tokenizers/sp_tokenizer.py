from sentencepiece import SentencePieceProcessor

from components.tokenizers.base_tokenizer import BaseTokenizer


class SPTokenizer(BaseTokenizer):
	''' SentencePiece tokenizer backed by a trained .model file.

	Properties:
		1. sp_model: SentencePieceProcessor  the loaded SentencePiece model.
	'''

	def __init__(self, sp_model_path: str,
				 bos_token_id: int | None = None,
				 eos_token_id: int | None = None,
				 pad_token_id: int | None = None,
				 max_length: int | None = None):
		''' Initialize the SentencePiece tokenizer.

		Args:
			1. sp_model_path: str  path to the trained SentencePiece .model file.
			2. bos_token_id: int | None  BOS token id for special token handling.
			3. eos_token_id: int | None  EOS token id for special token handling.
			4. pad_token_id: int | None  PAD token id for padding.
			5. max_length: int | None  max sequence length for truncation/padding.
		'''
		super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id,
						 pad_token_id=pad_token_id, max_length=max_length)
		self.sp_model = SentencePieceProcessor(model_file=sp_model_path)

	def _tokenize(self, text: str) -> list[int]:
		''' Encode text to token ids using SentencePiece.

		Args:
			1. text: str  raw input text.
		Returns:
			tokens: list[int]  encoded token ids.
		'''
		return self.sp_model.encode(text)

	def _detokenize(self, tokens: list[int]) -> str:
		''' Decode token ids back to text using SentencePiece.

		Args:
			1. tokens: list[int]  token ids to decode.
		Returns:
			text: str  decoded text.
		'''
		return self.sp_model.decode(tokens)
