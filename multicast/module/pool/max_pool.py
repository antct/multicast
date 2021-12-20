import torch
import torch.nn as nn
import numpy as np

class MaxPool(nn.Module):

    def __init__(self, kernel_size, segment_num=None):
        super().__init__()
        self.segment_num = segment_num
        if self.segment_num != None:
            self.mask_embedding = nn.Embedding(segment_num + 1, segment_num)
            self.mask_embedding.weight.data.copy_(torch.FloatTensor(np.concatenate([np.zeros((1, segment_num)), np.identity(segment_num)], axis=0)))
            self.mask_embedding.weight.requires_grad = False
            self._minus = -100
        self.pool = nn.MaxPool1d(kernel_size)

    def forward(self, x, mask=None):
        if mask is None or self.segment_num is None or self.segment_num == 1:
            x = x.transpose(1, 2)
            x = self.pool(x).squeeze(-1)
            return x
        else:
            mask = 1 - self.mask_embedding(mask).transpose(1, 2)
            x = x.transpose(1, 2)
            pool1 = self.pool(x + self._minus * mask[:, 0:1, :])
            pool2 = self.pool(x + self._minus * mask[:, 1:2, :])
            pool3 = self.pool(x + self._minus * mask[:, 2:3, :])
            x = torch.cat([pool1, pool2, pool3], 1)
            return  x
