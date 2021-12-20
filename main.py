import torch
import json
import os
import random
import multicast
import numpy as np
import logging

from multicast import encoder, model, framework
from tensorboardX import SummaryWriter
from config import args

# seed setting
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.device_count() > 0:
    torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic=True

# basic settings
if not os.path.exists('ckpt'):
    os.makedirs('ckpt')
if not os.path.exists('result'):
    os.makedirs('result')
if not os.path.exists('summary'):
    os.makedirs('summary')

model_name = args.model_name
# checkpoint save path
ckpt_path = 'ckpt/{}.pt'.format(model_name)

# pr result save path
p_path = 'result/{}_p.npy'.format(model_name)
r_path = 'result/{}_r.npy'.format(model_name)

# check data
rel2id = json.load(open('benchmark/nyt/nyt_rel2id.json'))
word2id = json.load(open('pretrain/glove/glove.6B.50d_word2id.json'))
word2vec = np.load('pretrain/glove/glove.6B.50d_mat.npy')

# sentence encoder
sentence_encoder = multicast.encoder.PCNNEncoder(
    token2id=word2id,
    max_length=120,
    word_size=50,
    position_size=5,
    hidden_size=230,
    blank_padding=True,
    kernel_size=3,
    padding_size=1,
    word2vec=word2vec,
    dropout=0.5
)

# define the model
model = multicast.model.BagAttention(
    sentence_encoder=sentence_encoder,
    rel2id=rel2id,
    hparams=args,
    mil=args.mil
)

writer = SummaryWriter('summary/{}'.format(model_name))
writer = None



# define the whole training framework
framework = multicast.framework.BagRE(
    model=model,
    writer=writer,
    train_path='benchmark/nyt/nyt_train.txt',
    val_path='benchmark/nyt/nyt_test.txt',
    test_path='benchmark/nyt/nyt_test.txt',
    ckpt=ckpt_path,
    batch_size=args.batch_size,
    max_epoch=args.max_epoch,
    pretrain_epoch=args.pretrain_epoch,
    pretrain_lr=args.pretrain_lr,
    lr=args.lr,
    lr_decay=args.lr_decay,
    lr_decay_epoch=args.lr_decay_epoch,
    lr_decay_rate=args.lr_decay_rate,
    lr_min=args.lr_min,
    weight_decay=args.weight_decay,
    opt=args.opt,
    bag_size=args.bag_size,
    loss_weight=args.loss_weight
)

framework.logger.info('framework: {}'.format(framework.name))
framework.logger.info('model: {}'.format(framework.model_name))
framework.logger.info('encoder: {}'.format(framework.encoder_name))
framework.logger.info('args: {}'.format(vars(args)))

if args.mode == 'train':
    best_auc = framework.train_model()

if args.mode == 'test':
    framework.load_state_dict(torch.load(ckpt_path)['state_dict'])
    result = framework.eval_model(framework.test_loader)
    print('AUC on test set: {}'.format(result['auc']))
