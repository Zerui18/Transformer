import abc

class Tokenizer(abc.ABC):

	def __init__(self, tokenize_args: dict = {}, detokenize_args: dict = {}):
		self.tokenize_args = tokenize_args
		self.detokenize_args = detokenize_args
		pass

	def tokenize(self, sentence: str) -> list:
		return self._tokenize(sentence, self.tokenize_args)
	
	def detokenize(self, tokens: list) -> str:
		return self._detokenize(tokens, self.detokenize_args)

	@abc.abstractmethod
	def _tokenize(self, sentence: str, tokenize_args: dict) -> list:
		pass
	
	@abc.abstractmethod
	def _detokenize(self, tokens: list, detokenize_args: dict) -> str:
		pass