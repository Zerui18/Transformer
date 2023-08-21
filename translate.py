import torch
import sentencepiece as sp
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
while True:
	# input src text
	text = input('Enter a sentence to translate: ').strip()
	src = torch.tensor(tokenizer.encode(text), dtype=torch.long)
	# translate, printing each new token to stdout
	print('Translation: ')
	tokens = []
	for token in model.translate_with_sampling(src, BOS_IDX, EOS_IDX, sampling='argmax', max_new_tokens=100):
		tokens.append(token)
		print(tokenizer.IdToPiece(token), end='')
	print()
	print('Full translation: ', tokenizer.decode(tokens))
	print()
	print('End of translation.')
