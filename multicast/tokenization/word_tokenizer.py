from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .utils import (load_vocab,
                   convert_to_unicode,
                   clean_text,
                   split_on_whitespace,
                   convert_by_vocab,
                   tokenize_chinese_chars)

class WordTokenizer(object):

    def __init__(self, vocab = None, unk_token="[UNK]"):
        self.vocab = load_vocab(vocab)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.unk_token = unk_token

    def tokenize(self, text):
        text = convert_to_unicode(text)
        text = clean_text(text)
        text = tokenize_chinese_chars(text)
        token_list = split_on_whitespace(text)
        return token_list

    def convert_tokens_to_ids(self, tokens, max_seq_length=None, blank_id=0, unk_id=1, uncased=True):
        return convert_by_vocab(self.vocab, tokens, max_seq_length, blank_id, unk_id, uncased=uncased)

    def convert_ids_to_tokens(self, ids):
        return convert_by_vocab(self.inv_vocab, ids)