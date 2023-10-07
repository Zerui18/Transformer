import torch
import numpy as np
import sentencepiece as sp
import argparse
from models.whisper import Whisper
from models.transformer import Transformer

UNK_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
PAD_IDX = 3

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=str, required=True, help='Model type to use', choices=['whisper', 'transformer'])
	parser.add_argument('--model-path', type=str, required=True, help='Path to model checkpoint')
	parser.add_argument('--tokenizer-path', type=str, required=True, help='Path to tokenizer model')
	return parser.parse_args()

def main():
	# parse args
	args = parse_args()

	# load model
	print('Loading model...')
	if args.model == 'whisper':
		model = Whisper.load_from_checkpoint(args.model_path).cuda()
	elif args.model == 'transformer':
		model = Transformer.load_from_checkpoint(args.model_path).cuda()

	# load tokenizer
	print('Loading tokenizer...')
	tokenizer = sp.SentencePieceProcessor(model_file=args.tokenizer_path)

	# translate
	print(f'Ready to {"translate" if args.model == "transformer" else "transcribe"}!')

	while True:
		if args.model == 'whisper':
			# input src mel
			mel = 'data/atis/mel_normalised/train/0k00f1ss.npy'
			#mel = input('Enter a path to a mel: ').strip()
			src = torch.tensor(np.load(mel).T, dtype=torch.float).cuda()
		elif args.model == 'transformer':
			# input src text
			text = input('Enter a sentence to translate: ').strip()
			src = torch.tensor(tokenizer.encode(text), dtype=torch.long)
		
		# translate, printing each new token to stdout
		print('Translation: ' if args.model == 'transformer' else 'Transcription: ')
		tokens = []
		for token in model.translate_with_sampling(src, BOS_IDX, EOS_IDX, sampling='argmax', max_new_tokens=100):
			tokens.append(token)
			print(tokenizer.IdToPiece(token).replace('▁', ' '), end='')
		print()
		print('End of translation.')

if __name__ == '__main__':
	main()