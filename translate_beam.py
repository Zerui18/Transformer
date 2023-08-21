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
model = Transformer.load_from_checkpoint('experiments/de-en-v1-multi30k/de-en-v1-sp-nb_6-multi30k/checkpoints/model-epoch=35-step=4000-val_loss=3.03.ckpt').cuda()
print('Loading tokenizer...')
tokenizer = sp.SentencePieceProcessor(model_file='data/multi30k/m_en_de.model')

# translate
print('Ready to translate.')

def stream_translate(stdscr, src: torch.Tensor):
	curses.noecho()
	curses.cbreak()
	beams = None
	lengths = [0] * 16
	for beam_bits in model.translate_with_beams(src, BOS_IDX, EOS_IDX, beam_width=16, max_new_tokens=20):
		if beams is None:
			beams = beam_bits
		else:
			beams = np.concatenate((beams, beam_bits), axis=-1)
		# print the beam_bits
		for i, bit in enumerate(beam_bits):
			if int(bit) == 0:
				continue
			text = tokenizer.IdToPiece(int(bit)).replace('▁', ' ')
			stdscr.addstr(i, lengths[i], text)
			lengths[i] += len(text)
		stdscr.refresh()
	stdscr.getkey()
while True:
	# input src text
	text = input('Enter a sentence to translate: ').strip()
	src = torch.tensor(tokenizer.encode(text), dtype=torch.long)
	curses.wrapper(stream_translate, src)