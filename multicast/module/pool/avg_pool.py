import torch
import torch.nn as nn
import numpy as np

class AvgPool(nn.Module):

    def __init__(self, kernel_size, segment_num=None):
        super().__init__()
        self.segment_num = segment_num
        if self.segment_num != None:
            self.mask_embedding = nn.Embedding(segment_num + 1, segment_num)
            self.mask_embedding.weight.data.copy_(torch.FloatTensor(np.concatenate([np.zeros(segment_num), np.identity(segment_num)], axis = 0)))
            self.mask_embedding.weight.requires_grad = False
        self.pool = nn.AvgPool1d(kernel_size)

    def forward(self, x, mask=None):
        if mask == None or self.segment_num == None or self.segment_num == 1:
            x = x.transpose(1, 2)
            x = self.pool(x).squeeze(-1)
            return x
        else:
            B, L, I_EMBED = x.size()[:2]
            mask = self.mask_embedding(mask).transpose(1, 2).unsqueeze(2)
            x = x.transpose(1, 2).unsqueeze(1)
            x = (x * mask).view([-1, I_EMBED, L])
            x = self.pool(x).squeeze(-1)
            x = x.view([B, -1]) - self._minus
            return x