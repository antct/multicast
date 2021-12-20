import torch
import torch.nn.functional as F
import numpy as np

from torch import nn
from torch.autograd import Variable, grad
from .base_model import BagRE


class BagAttention(BagRE):

    def __init__(self, sentence_encoder, rel2id, hparams, mil='att'):
        super().__init__()
        self.sentence_encoder = sentence_encoder
        self.rel2id = rel2id
        self.num_class = len(self.rel2id)
        self.fc = nn.Linear(self.sentence_encoder.hidden_size, self.num_class)
        self.softmax = nn.Softmax(-1)
        self.drop = nn.Dropout()
        self.kl = nn.KLDivLoss()
        self.criterion = nn.CrossEntropyLoss()
        self.mil = mil
        self.hparams = hparams
        self.id2rel = {}
        for rel, rel_id in rel2id.items():
            self.id2rel[rel_id] = rel

    def infer(self, bag):
        pass

    def _l2_normalize(self, d):
        size = len(d.size())
        d = d.numpy()
        if size == 2:
            d /= (np.sqrt(np.sum(d ** 2, axis=(1))).reshape((-1, 1)) + 1e-16)
        if size == 3:
            d /= (np.sqrt(np.sum(d ** 2, axis=(1, 2))).reshape((-1, 1, 1)) + 1e-16)
        return torch.from_numpy(d)

    def _kl_with_logit(self, q_logit, p_logit, mask=None):
        q = F.softmax(q_logit, dim=1)
        logq = F.log_softmax(q_logit, dim=1)
        logp = F.log_softmax(p_logit, dim=1)
        if mask is not None:
            total = mask.sum(dim=0)
            mask = torch.div(mask, total)
            qlogq = torch.mm((q * logq).sum(dim=1).unsqueeze(0), mask.unsqueeze(1))
            qlogp = torch.mm((q * logp).sum(dim=1).unsqueeze(0), mask.unsqueeze(1))
        else:
            qlogq = (q * logq).sum(dim=1).mean(dim=0)
            qlogp = (q * logp).sum(dim=1).mean(dim=0)
        return qlogq - qlogp

    def _cal_at_loss(self, bag_rep, label):
        at_logits = self.fc(bag_rep)
        at_loss = self.criterion(at_logits, label)
        rep_grad = grad(at_loss, bag_rep, retain_graph=True)
        r_adv = torch.FloatTensor(self.hparams.at_eps * self._l2_normalize(rep_grad[0].data.cpu()))
        r_adv = Variable(r_adv.cuda())
        at_logits = self.fc(bag_rep + r_adv.detach())
        self.zero_grad()
        at_loss = self.hparams.at_alpha * self.criterion(at_logits, label)
        return at_loss

    def _cal_vat_loss(self, token, pos1, pos2, mask=None, noisy_mask=None, att_mask=None):
        vat_loss = None
        noisy_mask = noisy_mask.long()
        noisy_mask = torch.nonzero(noisy_mask).view(-1)
        if noisy_mask.size(0) != 0:
            token = torch.index_select(token, 0, noisy_mask)
            pos1 = torch.index_select(pos1, 0, noisy_mask)
            pos2 = torch.index_select(pos2, 0, noisy_mask)
            if mask is not None:
                mask = torch.index_select(mask, 0, noisy_mask)
            if att_mask is not None:
                att_mask = torch.index_select(att_mask, 0, noisy_mask)
            batch_size, max_length = token.size()
            embed_size = (batch_size, max_length,
                          self.sentence_encoder.word_size)
            d = torch.Tensor(torch.Size(embed_size)).normal_()
            if mask is not None:
                rep = self.sentence_encoder(token, pos1, pos2, mask)
            else:
                rep = self.sentence_encoder(token, pos1, pos2)
            y_f = self.fc(rep)
            for _ in range(self.hparams.vat_iter):
                d = self.hparams.vat_xi * self._l2_normalize(d)
                d = Variable(d.cuda(), requires_grad=True)
                if mask is not None:
                    rep = self.sentence_encoder(token, pos1, pos2, mask, d=d)
                else:
                    rep = self.sentence_encoder(token, pos1, pos2, d=d)
                y_n = self.fc(rep)
                delta_kl = self._kl_with_logit(y_f.detach(), y_n)
                delta_kl.backward()
                d = d.grad.data.clone().cpu()
                self.zero_grad()
            d = self._l2_normalize(d)
            d = Variable(d.cuda())
            r_adv = self.hparams.vat_eps * d
            if mask is not None:
                rep = self.sentence_encoder(token, pos1, pos2, mask, d=r_adv.detach())
            else:
                rep = self.sentence_encoder(token, pos1, pos2, d=r_adv.detach())
            y_e = self.fc(rep)
            if att_mask is not None:
                vat_loss = self.hparams.vat_alpha * self._kl_with_logit(y_f.detach(), y_e, mask=att_mask.detach())
            else:
                vat_loss = self.hparams.vat_alpha * self._kl_with_logit(y_f.detach(), y_e)
        return vat_loss

    def forward(self, label, scope, token, pos1, pos2, mask=None, train=True, bag_size=0):
        if bag_size > 0:
            token = token.view(-1, token.size(-1))
            pos1 = pos1.view(-1, pos1.size(-1))
            pos2 = pos2.view(-1, pos2.size(-1))
            if mask is not None: mask = mask.view(-1, mask.size(-1))
        else:
            begin, end = scope[0][0], scope[-1][1]
            token = token[:, begin:end, :].view(-1, token.size(-1))
            pos1 = pos1[:, begin:end, :].view(-1, pos1.size(-1))
            pos2 = pos2[:, begin:end, :].view(-1, pos2.size(-1))
            if mask is not None: mask = mask[:, begin:end, :].view(-1, mask.size(-1))
            scope = torch.sub(scope, torch.zeros_like(scope).fill_(begin))

        if mask is not None:
            rep = self.sentence_encoder(token, pos1, pos2, mask)  # (nsum, H)
        else:
            rep = self.sentence_encoder(token, pos1, pos2)  # (nsum, H)

        items = []
        noisy_mask, att_mask = None, None
        if train:
            if bag_size == 0:
                bag_rep = []
                query = torch.zeros((rep.size(0))).long()
                if torch.cuda.is_available():
                    query = query.cuda()
                for i in range(len(scope)):
                    query[scope[i][0]:scope[i][1]] = label[i]
                att_mat = self.fc.weight.data[query]  # (nsum, H)
                att_score = (rep * att_mat).sum(-1)  # (nsum)
                for i in range(len(scope)):
                    bag_mat = rep[scope[i][0]:scope[i][1]]  # (n, H)
                    softmax_att_score = self.softmax(
                        att_score[scope[i][0]:scope[i][1]])  # (n)
                    noisy_idx = torch.le(
                        softmax_att_score, self.hparams.vat_threshold)
                    noisy_mask = noisy_idx if noisy_mask is None else torch.cat(
                        [noisy_mask, noisy_idx], 0)
                    att_mask = softmax_att_score if att_mask is None else torch.cat(
                        [att_mask, softmax_att_score], 0)
                    if self.mil == 'one':
                        one_index = softmax_att_score.argmax(1).cpu()
                        one_att_score = torch.zeros(softmax_att_score.shape).scatter(
                            1, one_index.unsqueeze(1), 1.0).cuda()
                        softmax_att_score = one_att_score
                    bag_rep.append((softmax_att_score.unsqueeze(-1) * bag_mat).sum(0))
                bag_rep = torch.stack(bag_rep, 0)  # (B, H)
                bag_logits = self.fc(self.drop(bag_rep))
            else:
                batch_size = label.size(0)
                query = label.unsqueeze(1)  # (B, 1)
                att_mat = self.fc.weight.data[query]  # (B, 1, H)
                rep = rep.view(batch_size, bag_size, -1)
                att_score = (rep * att_mat).sum(-1)  # (B, bag)
                softmax_att_score = self.softmax(att_score)  # (B, bag)
                noisy_idx = torch.le(softmax_att_score, self.hparams.vat_threshold)
                noisy_mask = noisy_idx.view(-1)
                att_mask = softmax_att_score
                att_mask = torch.sub(torch.ones_like(att_mask), att_mask)
                if self.mil == 'one':
                    one_index = softmax_att_score.argmax(1).cpu()
                    one_att_score = torch.zeros(softmax_att_score.shape).scatter(1, one_index.unsqueeze(1), 1.0).cuda()
                    softmax_att_score = one_att_score
                # (B, bag, 1) * (B, bag, H) -> (B, bag, H) -> (B, H)
                bag_rep = (softmax_att_score.unsqueeze(-1) * rep).sum(1)
                bag_logits = self.fc(self.drop(bag_rep))  # (B, N)
            items.append(self._cal_vat_loss(token, pos1, pos2, mask, noisy_mask))
            items.append(self._cal_at_loss(bag_rep, label))
        else:
            if bag_size == 0:
                bag_logits = []
                # (nsum, H) * (H, N) -> (nsum, N)
                att_score = torch.matmul(
                    rep, self.fc.weight.data.transpose(0, 1))
                for i in range(len(scope)):
                    bag_mat = rep[scope[i][0]:scope[i][1]]  # (n, H)
                    softmax_att_score = self.softmax(att_score[scope[i][0]:scope[i][1]].transpose(0, 1))  # (N, n) num_labels
                    rep_for_each_rel = torch.matmul(softmax_att_score, bag_mat)  # (N, n) * (n, H) -> (N, H)
                    logit_for_each_rel = self.softmax(self.fc(rep_for_each_rel))  # ((each rel)N, (logit)N)
                    logit_for_each_rel = logit_for_each_rel.diag()  # (N)
                    bag_logits.append(logit_for_each_rel)
                bag_logits = torch.stack(bag_logits, 0)
            else:
                batch_size = rep.size(0) // bag_size
                # (nsum, H) * (H, N) -> (nsum, N)
                att_score = torch.matmul(rep, self.fc.weight.data.transpose(0, 1))
                att_score = att_score.view(batch_size, bag_size, -1) # (B, bag, N)
                rep = rep.view(batch_size, bag_size, -1) # (B, bag, H)
                softmax_att_score = self.softmax(att_score.transpose(1, 2)) # (B, N, (softmax)bag)
                # (B, N, bag) * (B, bag, H) -> (B, N, H)
                rep_for_each_rel = torch.matmul(softmax_att_score, rep)
                bag_logits = self.softmax(self.fc(rep_for_each_rel)).diagonal(dim1=1, dim2=2) # (B, (each rel)N)
        items.append(bag_logits)
        items = items if len(items) > 1 else items[0]
        return items