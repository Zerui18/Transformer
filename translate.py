from model import TransformerModelStockLN

def translate(args):
	# load model
	model = TransformerModelStockLN.load_from_checkpoint(args.checkpoint_path).cuda()
	while True:
		# input src text
		text = input('Enter a sentence to translate: ').strip()
		# translate, printing each new token to stdout
		print('Translation: ')
		for token in model.translate(text):
			print(token, end='')
		print()
		print('End of translation.')
