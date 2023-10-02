import torch
import numpy as np
import sentencepiece as sp
import curses
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

	def stream_translate(stdscr, src: torch.Tensor):
		curses.noecho()
		curses.cbreak()
		beams = None
		lengths = [0] * 16
		for beam_bits in model.translate_with_beams(src, BOS_IDX, EOS_IDX, beam_width=8, max_new_tokens=20):
			if beams is None:
				beams = beam_bits
			else:
				beams = np.concatenate((beams, beam_bits), axis=-1)
			# print the beam_bits
			for i, bit in enumerate(beam_bits):
				text = tokenizer.IdToPiece(int(bit)).replace('▁', ' ')
				stdscr.addstr(i * 2, lengths[i], text)
				lengths[i] += len(text)
			stdscr.refresh()
		stdscr.getkey()
	while True:
		if args.model == 'whisper':
			# input src mel
			mel = 'data/atis/mel_normalised/test/x11037ss.npy'
			#mel = input('Enter a path to a mel: ').strip()
			src = torch.tensor(np.load(mel).T, dtype=torch.float).cuda()
		elif args.model == 'transformer':
			# input src text
			text = input('Enter a sentence to translate: ').strip()
			src = torch.tensor(tokenizer.encode(text), dtype=torch.long)
		curses.wrapper(stream_translate, src)

if __name__ == '__main__':
	main()