import torch
from torch.utils.data import Dataset
import sentencepiece as sp
from tqdm import tqdm
import pandas as pd

class TextGenerationDataset(Dataset):

	def __init__(self, file, block_len=512, tokenizer=None):
		super().__init__()
		''' Create a text generation dataset with the source file containing the plain text. '''
		self.block_len = block_len
		print('Initializing Dataset')
		print(f'Reading {file}...')
		with open(file) as f:
			self._raw_text = f.read()
		print('Tokenizing text...')
		tokenizer = tokenizer or (lambda x:x)
		self._vectorized_text = tokenizer(self._raw_text)
		self._vectorized_text = torch.tensor(self._vectorized_text, dtype=torch.long)

	def __getitem__(self, idx):
		x = self._vectorized_text[idx: idx+self.block_len]
		y = self._vectorized_text[idx+1: idx+self.block_len+1]
		return x, y

	def __len__(self):
		return self._vectorized_text.shape[0] - self.block_len
	
class TranslationDataset(Dataset):

	def __init__(self, train_src_file: str, train_dst_file: str, sp_model: str, block_size: int):
		super().__init__()
		print('Reading input files...')
		with open(train_src_file) as f:
			src_lines = list(f)
		with open(train_dst_file) as f:
			dst_lines = list(f)
		self.df = pd.DataFrame({ 'src' : src_lines , 'dst' : dst_lines })
		self.tokenizer = sp.SentencePieceProcessor(model_name=sp_model)
		self.block_len = block_size

	# @staticmethod
	# def build_dataset(ds_name: str, train_src_file: str, train_dst_file: str, sp_model: str, block_len: int):
	# 	print('Reading input files...')
	# 	with open(train_src_file) as f:
	# 		src_lines = list(f)
	# 	with open(train_dst_file) as f:
	# 		dst_lines = list(f)
	# 	print('Tokenizing input files...')
	# 	tokenizer = sp.SentencePieceProcessor(model_file=sp_model)
	# 	src_lines = [tokenizer.encode(l) for l in tqdm(src_lines)]
	# 	dst_lines = [tokenizer.encode(l) for l in tqdm(dst_lines)]
	# 	print('Writing cached dataset...')
	# 	df = pd.DataFrame({ 'src' : src_lines , 'dst' : dst_lines })
	# 	df.to_pickle(f'{ds_name}.cache.p')
	# 	print('Dataset cached as:', f'{ds_name}.cache.p')
	# 	return TranslationDataset(df, block_len)
	
	def __len__(self):
		return len(self.df)
	
	def __getitem__(self, idx):
		row = self.df.iloc[idx]
		src, dst = row.src, row.dst
		# enable bpe dropout regularization
		src = self.tokenizer.encode(src, enable_sampling=True, alpha=0.1, nbest_size=-1)
		dst = self.tokenizer.encode(dst, enable_sampling=True, alpha=0.1, nbest_size=-1)
		return torch.tensor(src), torch.tensor(dst)