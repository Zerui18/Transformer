import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from toknizers import Tokenizer

UNK_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
PAD_IDX = 3

def get_bleu_score(pred: list, reference: list, tokenizer: Tokenizer) -> float:
	'''
	Calculate BLEU-4 score of a single prediction and reference pair by detokenizing and using NLTK.

	Args:
		`pred`: List<Long>[T1] predicted tokens.
		`reference`: List<Long>[T2] reference tokens.
	
	Returns:
		`score`: float BLEU-4 score in [0, 1].
	'''
	# convert to numpy arrays
	pred = np.array(pred, dtype=int)
	reference = np.array(reference, dtype=int)
	# clean pred and reference
	# remove all <unk> and <pad> tokens
	pred = pred[pred != UNK_IDX]
	pred = pred[pred != PAD_IDX]
	reference = reference[reference != UNK_IDX]
	reference = reference[reference != PAD_IDX]
	#print('cleaned predictio:', pred)
	#print('cleaned reference:', reference)
	# detokenize
	pred = tokenizer.detokenize([int(t) for t in pred])
	reference = tokenizer.detokenize([int(t) for t in reference])
	#print('detokenized predictio:', pred)
	#print('detokenized reference:', reference)
	# re-tokenize
	pred = tokenizer.tokenize(pred)
	reference = tokenizer.tokenize(reference)
	#print('re-tokenized predictio:', pred)
	#print('re-tokenized reference:', reference)
	# calculate BLEU-4 score
	return sentence_bleu([reference], pred)
