import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_encoder import BaseEncoder

class PCNNEncoder(BaseEncoder):

    def __init__(self,
                 token2id,
                 max_length=128,
                 hidden_size=230,
                 word_size=50,
                 position_size=5,
                 blank_padding=True,
                 word2vec=None,
                 kernel_size=3,
                 padding_size=1,
                 dropout=0.0,
                 activation_function=F.relu):
        super().__init__(token2id, max_length, hidden_size, word_size, position_size, blank_padding, word2vec)
        self.drop = nn.Dropout(dropout)
        self.kernel_size = kernel_size
        self.padding_size = padding_size
        self.act = activation_function
        self.conv = nn.Conv1d(self.input_size, self.hidden_size, self.kernel_size, padding=self.padding_size)
        self.pool = nn.MaxPool1d(self.max_length)
        self.mask_embedding = nn.Embedding(4, 3)
        self.mask_embedding.weight.data.copy_(torch.FloatTensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        self.mask_embedding.weight.requires_grad = False
        self._minus = -100
        self.hidden_size *= 3

    def forward(self, token, pos1, pos2, mask, d=None):
        if len(token.size()) != 2 or token.size() != pos1.size() or token.size() != pos2.size():
            raise Exception("size of token, pos1 ans pos2 should be (B, L)")
        emb = self.word_embedding(token) + d if d is not None else self.word_embedding(token)
        x = torch.cat([emb, self.pos1_embedding(pos1), self.pos2_embedding(pos2)], 2)
        x = x.transpose(1, 2)
        x = self.conv(x)
        mask = 1 - self.mask_embedding(mask).transpose(1, 2)
        pool1 = self.pool(self.act(x + self._minus * mask[:, 0:1, :]))
        pool2 = self.pool(self.act(x + self._minus * mask[:, 1:2, :]))
        pool3 = self.pool(self.act(x + self._minus * mask[:, 2:3, :]))
        x = torch.cat([pool1, pool2, pool3], 1)
        x = x.squeeze(2)
        x = self.drop(x)
        return x

    def tokenize(self, item):
        return super().tokenize(item, mask=True)
