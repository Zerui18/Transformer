import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k
from torch.nn.utils.rnn import pad_sequence
from typing import Iterable, List
from dataclasses import dataclass
from models.transformer import TransformerInputBatch
from .base import BaseDataset

''' Currently only de -> en is supported as lots of stuff are hardcoded. '''

# Define special symbols and indices
UNK_IDX, BOS_IDX, EOS_IDX, PAD_IDX = 0, 1, 2, 3

# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

# helper function to yield list of tokens
def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    language_index = {'de': 0, 'en': 1}

    for data_sample in data_iter:
        yield TranslationDatasetSpacyMulti30K.token_transform[language](data_sample[language_index[language]])

@dataclass
class TranslationDatasetSpacyMulti30KConfig:
  src_language: str
  tgt_language: str
  split: str

class TranslationDatasetSpacyMulti30K(BaseDataset):

    _INIT_RESOURCES_DONE = False
    token_transform = None
    vocab_transform = None
    text_transform = None

    def __init__(self, config: TranslationDatasetSpacyMulti30KConfig):
        TranslationDatasetSpacyMulti30K._init_resources()
        super().__init__()
        self.src_lang = config.src_language
        self.tgt_lang = config.tgt_language
        self.split = config.split
        self.dataset = list(Multi30k(split=self.split, language_pair=(self.src_lang, self.tgt_lang)))

    @staticmethod
    def _init_resources():
        if TranslationDatasetSpacyMulti30K._INIT_RESOURCES_DONE:
            return

        # We need to modify the URLs for the dataset since the links to the original dataset are broken
        # Refer to https://github.com/pytorch/text/issues/1756#issuecomment-1163664163 for more info
        multi30k.URL["train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
        multi30k.URL["valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"

        # Place-holders
        token_transform = {}
        vocab_transform = {}

        SRC_LANG = 'de'
        TGT_LANG = 'en'

        token_transform[SRC_LANG] = get_tokenizer('spacy', language='de_core_news_sm')
        token_transform[TGT_LANG] = get_tokenizer('spacy', language='en_core_web_sm')

        TranslationDatasetSpacyMulti30K.token_transform = token_transform

        # Make sure the tokens are in order of their indices to properly insert them in vocab
        special_symbols = ['<unk>', '<bos>', '<eos>', '<pad>']

        for ln in [SRC_LANG, TGT_LANG]:
            # Training data Iterator
            train_iter = Multi30k(split='train', language_pair=(SRC_LANG, TGT_LANG))
            # Create torchtext's Vocab object
            vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                            min_freq=1,
                                                            specials=special_symbols,
                                                            special_first=True)
            
        # Set UNK_IDX as the default index. This index is returned when the token is not found.
        # If not set, it throws RuntimeError when the queried token is not found in the Vocabulary.
        for ln in [SRC_LANG, TGT_LANG]:
            vocab_transform[ln].set_default_index(UNK_IDX)

        TranslationDatasetSpacyMulti30K.vocab_transform = vocab_transform

        # src and tgt language text transforms to convert raw strings into tensors indices
        text_transform = {}
        for ln in [SRC_LANG, TGT_LANG]:
            text_transform[ln] = sequential_transforms(token_transform[ln], #Tokenization
                                                    vocab_transform[ln], #Numericalization
                                                    tensor_transform) # Add BOS/EOS and create tensor
        
        TranslationDatasetSpacyMulti30K.text_transform = text_transform
        TranslationDatasetSpacyMulti30K._INIT_RESOURCES_DONE = True

        # print vocab sizes
        print(f"Vocab size for {SRC_LANG}: {len(vocab_transform[SRC_LANG])}")
        print(f"Vocab size for {TGT_LANG}: {len(vocab_transform[TGT_LANG])}")

    def __len__(self):
        if self.split == 'train':
            return 29000
        elif self.split == 'valid':
            return 1014
        
    def __getitem__(self, idx):
        src, dst = self.dataset[idx]
        src = TranslationDatasetSpacyMulti30K.text_transform['de'](src.rstrip("\n"))
        dst = TranslationDatasetSpacyMulti30K.text_transform['en'](dst.rstrip("\n"))
        return src, dst

    @staticmethod
    def get_collate_function():
        def collate_fn(batch):
            src_batch, tgt_batch = zip(*batch)
            src_batch = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
            tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)
            return TransformerInputBatch(src_batch, tgt_batch[:, :-1].clone(), src_batch != PAD_IDX, tgt_batch[:, :-1] != PAD_IDX, tgt_batch[:, 1:].clone())
        return collate_fn