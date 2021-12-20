import logging
import os
import sys

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        if self.count == 0: return str(self.val)
        return '%.4f (%.4f)' % (self.val, self.avg)


class Logger(object):

    def __init__(self):
        pass

    @staticmethod
    def get(prefix, save_dir='./logs'):
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        logger = logging.getLogger('multicast')
        if not os.path.exists(save_dir): os.mkdir(save_dir)
        file_handler = logging.FileHandler(save_dir+'/'+prefix+'.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)
        return logger