import torch
import torch.utils.data as data
import numpy as np
import random
import sklearn.metrics


class BagREDataset(data.Dataset):

    def __init__(self, path, rel2id, tokenizer, entpair_as_bag=False, bag_size=0):
        super().__init__()
        self.tokenizer = tokenizer
        self.rel2id = rel2id
        self.entpair_as_bag = entpair_as_bag
        self.bag_size = bag_size
        self.data = open(path, 'r').read().strip().splitlines()
        self.data = [eval(i) for i in self.data]
        self.weight = np.zeros((len(self.rel2id)), dtype=np.float32)
        self.bag_scope = []
        self.name2id = {}
        self.bag_name = []
        self.facts = {}
        for idx, item in enumerate(self.data):
            fact = (item['h']['id'], item['t']['id'], item['relation'])
            if item['relation'] != 'NA':
                self.facts[fact] = 1
            if entpair_as_bag:
                name = (item['h']['id'], item['t']['id'])
            else:
                name = fact
            if name not in self.name2id:
                self.name2id[name] = len(self.name2id)
                self.bag_scope.append([])
                self.bag_name.append(name)
            self.bag_scope[self.name2id[name]].append(idx)
            self.weight[self.rel2id[item['relation']]] += 1.0
        self.weight = 1.0 / (self.weight ** 0.05 + 1e-6)
        self.weight = torch.from_numpy(self.weight)

    def __len__(self):
        return len(self.bag_scope)

    def __getitem__(self, index):
        bag = self.bag_scope[index]
        if self.bag_size > 0:
            if self.bag_size <= len(bag):
                resize_bag = random.sample(bag, self.bag_size)
            else:
                resize_bag = bag + list(np.random.choice(bag, self.bag_size - len(bag)))
            bag = resize_bag
        seqs = None
        rel = self.rel2id[self.data[bag[0]]['relation']]
        for sent_id in bag:
            item = self.data[sent_id]
            seq = list(self.tokenizer(item))
            if seqs is None:
                seqs = [[] for _ in range(len(seq))]
            for i in range(len(seq)):
                seqs[i].append(seq[i])
        seqs = [torch.cat(seq, 0) for seq in seqs]
        return [rel, self.bag_name[index], len(bag), bag] + seqs

    def collate_fn(data):
        data = list(zip(*data))
        label, bag_name, count, bag = data[:4]
        seqs = data[4:]
        for i in range(len(seqs)):
            seqs[i] = torch.cat(seqs[i], 0)
            seqs[i] = seqs[i].expand((torch.cuda.device_count(), ) + seqs[i].size())
        scope = []
        start = 0
        for c in count:
            scope.append((start, start + c))
            start += c
        assert(start == seqs[0].size(1))
        scope = torch.tensor(scope).long()
        label = torch.tensor(label).long()
        return [label, bag_name, scope, bag] + seqs

    def collate_bag_size_fn(data):
        data = list(zip(*data))
        label, bag_name, count, bag = data[:4]
        seqs = data[4:]
        for i in range(len(seqs)):
            seqs[i] = torch.stack(seqs[i], 0)
        scope = []
        start = 0
        for c in count:
            scope.append((start, start + c))
            start += c
        label = torch.tensor(label).long()
        return [label, bag_name, scope, bag] + seqs

    def eval(self, pred_result):
        sorted_pred_result = sorted(pred_result, key=lambda x: x['score'], reverse=True)
        prec, rec = [], []
        correct = 0
        total = len(self.facts)
        for i, item in enumerate(sorted_pred_result):
            if (item['entpair'][0], item['entpair'][1], item['relation']) in self.facts:
                correct += 1
            prec.append(float(correct) / float(i + 1))
            rec.append(float(correct) / float(total))
        auc = sklearn.metrics.auc(x=rec, y=prec)
        np_prec = np.array(prec)
        np_rec = np.array(rec)
        f1 = (2 * np_prec * np_rec / (np_prec + np_rec + 1e-20)).max()
        mean_prec = np_prec.mean()
        return {'prec': np_prec, 'rec': np_rec, 'mean_prec': mean_prec, 'f1': f1, 'auc': auc}


def BagRELoader(path, rel2id, tokenizer, batch_size,
                shuffle, entpair_as_bag=False, bag_size=0,
                num_workers=8, collate_fn=BagREDataset.collate_fn):
    if bag_size == 0:
        collate_fn = BagREDataset.collate_fn
    else:
        collate_fn = BagREDataset.collate_bag_size_fn
    dataset = BagREDataset(
        path=path,
        rel2id=rel2id,
        tokenizer=tokenizer,
        entpair_as_bag=entpair_as_bag,
        bag_size=bag_size
    )
    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return data_loader
