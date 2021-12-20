import torch
import time

from torch import nn, optim
from tqdm import tqdm
from .data_loader import BagRELoader
from .utils import AverageMeter, Logger


class BagRE(nn.Module):

    def __init__(self,
                 model,
                 writer,
                 train_path,
                 val_path,
                 test_path,
                 ckpt,
                 batch_size=32,
                 max_epoch=100,
                 pretrain_epoch=40,
                 pretrain_lr=0.5,
                 lr=0.1,
                 lr_decay=False,
                 lr_decay_epoch=20,
                 lr_decay_rate=0.1,
                 lr_min=0.01,
                 weight_decay=1e-5,
                 opt='sgd',
                 bag_size=0,
                 loss_weight=False):

        super().__init__()
        self.max_epoch = max_epoch
        self.bag_size = bag_size
        self.train_loader = BagRELoader(
            path=train_path,
            rel2id=model.rel2id,
            tokenizer=model.sentence_encoder.tokenize,
            batch_size=batch_size,
            shuffle=True,
            bag_size=bag_size,
            entpair_as_bag=False
        )
        self.val_loader = BagRELoader(
            path=val_path,
            rel2id=model.rel2id,
            tokenizer=model.sentence_encoder.tokenize,
            batch_size=batch_size,
            shuffle=False,
            bag_size=0,
            entpair_as_bag=True
        )
        self.test_loader = BagRELoader(
            path=test_path,
            rel2id=model.rel2id,
            tokenizer=model.sentence_encoder.tokenize,
            batch_size=batch_size,
            shuffle=False,
            bag_size=0,
            entpair_as_bag=True
        )
        self.model = nn.DataParallel(model)
        self.writer = writer
        self.criterion = nn.CrossEntropyLoss(weight=self.train_loader.dataset.weight if loss_weight else None)
        self.pretrain_epoch = pretrain_epoch
        self.pretrain_lr = pretrain_lr
        self.lr = lr
        self.lr_min = lr_min
        self.lr_decay = lr_decay
        self.opt = opt
        params = self.model.parameters()
        if opt == 'sgd':
            self.optimizer = optim.SGD(params, lr, weight_decay=weight_decay)
        elif opt == 'adam':
            self.optimizer = optim.Adam(params, lr, weight_decay=weight_decay)
        elif opt == 'adadelta':
            self.optimizer = optim.Adadelta(params, lr=1.0, rho=0.95, eps=1e-06, weight_decay=weight_decay)
        else:
            pass
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, lr_decay_epoch, lr_decay_rate)
        if torch.cuda.is_available(): self.cuda()
        self.ckpt = ckpt
        self.global_steps = 0
        self.name = self.__class__.__name__
        self.model_name = self.model.module.__class__.__name__
        self.encoder_name = self.model.module.sentence_encoder.__class__.__name__
        prefix = '-'.join([self.name, self.model_name, self.encoder_name, str(time.time())])
        self.logger = Logger.get(prefix=prefix)


    def train_model(self):
        best_auc, best_f1, run_time = 0., 0., 0.
        for epoch in range(1, self.max_epoch+1):
            self.model.train()
            if epoch == 1:
                for g in self.optimizer.param_groups:
                    g['lr'] = self.pretrain_lr
            if epoch == self.pretrain_epoch + 1:
                for g in self.optimizer.param_groups:
                    g['lr'] = self.lr
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.lr_decay and current_lr >= self.lr_min:
                self.scheduler.step()
            if self.writer is not None:
                self.writer.add_scalar('train/lr', current_lr, epoch)
            self.logger.info("epoch {} train lr {}".format(epoch, current_lr))
            avg_mil_loss, avg_at_loss, avg_vat_loss = [AverageMeter()] * 3
            avg_acc, avg_pos_acc = AverageMeter(), AverageMeter()
            start_time = time.time()
            t = tqdm(self.train_loader)
            for _, data in enumerate(t):
                if torch.cuda.is_available():
                    data = [i.cuda() if torch.is_tensor(i) else i for i in data]
                self.global_steps += 1
                label, bag_name, scope, bag, args = data[0], data[1], data[2], data[3], data[4:]
                vat_loss, at_loss, logits = self.model(label, scope, *args, bag_size=self.bag_size)
                mil_loss = self.criterion(logits, label)
                score, pred = logits.max(-1)
                acc = float((pred == label).long().sum()) / label.size(0)
                pos_total = (label != 0).long().sum()
                pos_correct = ((pred == label).long() * (label != 0).long()).sum()
                if pos_total > 0:
                    pos_acc = float(pos_correct) / float(pos_total)
                else:
                    pos_acc = 0
                if vat_loss is not None: avg_vat_loss.update(vat_loss.item(), 1)
                if at_loss is not None: avg_at_loss.update(at_loss.item(), 1)
                avg_mil_loss.update(mil_loss.item(), 1)
                avg_acc.update(acc, 1)
                avg_pos_acc.update(pos_acc, 1)
                t.set_postfix(mil_loss=avg_mil_loss.avg, vat_loss=avg_vat_loss.avg, at_loss=avg_at_loss.avg, acc=avg_acc.avg, pos_acc=avg_pos_acc.avg)
                loss = mil_loss
                if vat_loss is not None: loss += vat_loss
                if at_loss is not None: loss += at_loss
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 5.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
            end_time = time.time()
            epoch_time = end_time - start_time
            run_time += epoch_time
            if self.writer is not None:
                self.writer.add_scalar('train/at_loss', avg_at_loss.avg, epoch)
                self.writer.add_scalar('train/vat_loss', avg_vat_loss.avg, epoch)
                self.writer.add_scalar('train/mil_loss', avg_mil_loss.avg, epoch)
                self.writer.add_scalar('train/acc', avg_acc.avg, epoch)
                self.writer.add_scalar('train/pos_acc', avg_pos_acc.avg, epoch)
                self.writer.add_scalar('train/run_time', run_time, epoch)
            self.logger.info("epoch_time: %ds avg_time: %ds run_time: %ds" % (epoch_time, run_time / epoch, run_time))
            self.logger.info("epoch {} val".format(epoch))
            result = self.eval_model(self.val_loader)
            self.logger.info("auc: %.4f auc@best: %.4f " % (result['auc'], best_auc))
            self.logger.info("f1: %.4f f1@best: %.4f" % (result['f1'], best_f1))
            p = result['prec']
            self.logger.info("p@100: %.4f p@200: %.4f p@300: %.4f p@m: %.4f" % (p[100], p[200], p[300], (p[100] + p[200] + p[300]) / 3))
            if self.writer is not None:
                self.writer.add_scalar('eval/loss', result['loss'], epoch)
                self.writer.add_scalar('eval/auc', result['auc'], epoch)
                self.writer.add_scalar('eval/f1', result['f1'], epoch)
                self.writer.add_scalar('eval/p_100', p[100], epoch)
                self.writer.add_scalar('eval/p_200', p[200], epoch)
                self.writer.add_scalar('eval/p_300', p[300], epoch)
                self.writer.add_scalar('eval/p_m', (p[100] + p[200] + p[300]) / 3, epoch)
            if result['auc'] > best_auc:
                self.logger.info("best ckpt and saved!")
                torch.save({'state_dict': self.model.module.state_dict()}, self.ckpt)
                best_auc = result['auc']
                best_f1 = result['f1']
            else:
                self.logger.info("no improvement, skip!")
        self.logger.info("best auc on val set: %f" % (best_auc))
        return best_auc


    def eval_model(self, eval_loader):
        self.model.eval()
        eval_losses = []
        pred_result = []
        with torch.no_grad():
            t = tqdm(eval_loader)
            for _, data in enumerate(t):
                if torch.cuda.is_available():
                    data = [i.cuda() if torch.is_tensor(i) else i for i in data]
                label, bag_name, scope, bag, args = data[0], data[1], data[2], data[3], data[4:]
                logits = self.model(None, scope, *args, train=False, bag_size=0)
                loss = self.criterion(logits, label)
                eval_losses.append(loss.item())
                logits = logits.cpu().numpy()
                for i in range(len(logits)):
                    for relid in range(self.model.module.num_class):
                        if self.model.module.id2rel[relid] != 'NA':
                            pred_result.append({
                                'entpair': bag_name[i][:2],
                                'relation': self.model.module.id2rel[relid],
                                'score': logits[i][relid].item()
                            })
            result = eval_loader.dataset.eval(pred_result)
        result['loss'] = sum(eval_losses) / len(eval_losses)
        return result


    def load_state_dict(self, state_dict):
        self.model.module.load_state_dict(state_dict)