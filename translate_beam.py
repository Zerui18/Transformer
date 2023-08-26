import torch
import numpy as np
import sentencepiece as sp
import curses
from models.transformer import Transformer

UNK_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
PAD_IDX = 3

# load model
print('Loading model...')
model = Transformer.load_from_checkpoint('experiments/en-fr-v1/en-fr-v1/checkpoints/model-epoch=5-step=1016000-val_loss=3.61.ckpt').cuda()
print('Loading tokenizer...')
tokenizer = sp.SentencePieceProcessor(model_file='data/un/processed/undoc.2000.fr-en.model')

# translate
print('Ready to translate.')

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
	# input src text
	text = input('Enter a sentence to translate: ').strip()
	src = torch.tensor(tokenizer.encode(text), dtype=torch.long)
	curses.wrapper(stream_translate, src)
