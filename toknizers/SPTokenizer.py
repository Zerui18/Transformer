from .tokenizer import Tokenizer
from sentencepiece import SentencePieceProcessor

class SPTokenizer(Tokenizer):

	def __init__(self, sp_model_path: str, tokenize_args: dict = {}, detokenize_args: dict = {}):
		self.sp_model = SentencePieceProcessor(model_file=sp_model_path)
		super().__init__(tokenize_args, detokenize_args)

	def _tokenize(self, sentence: str, tokenize_args: dict) -> list:
		return self.sp_model.encode(sentence, **tokenize_args)
	
	def _detokenize(self, tokens: list, detokenize_args: dict) -> str:
		return self.sp_model.decode(tokens, **detokenize_args)