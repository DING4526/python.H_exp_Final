import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, context
from mindspore.train.callback import Callback
import matplotlib.pyplot as plt
import time
import os

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args, printout=True):
    lr_adjust = {}
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.8 ** ((epoch - 3) // 1))}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate * 0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate * 0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate * 0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate * 0.1}
    #elif args.lradj == 'TST':
    #    lr_adjust = {epoch: scheduler.get_last_lr()[0]}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.parameters:
            param_group['learning_rate'] = lr
        if printout:
            print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    """
    早停机制，用于在验证损失不再下降时提前终止训练。
    """

    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        """
        保存模型检查点，如果验证损失有所下降。
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # 使用 MindSpore 的保存方法
        ms.save_checkpoint(model, os.path.join(path, 'checkpoint.ckpt'))
        self.val_loss_min = val_loss


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler:

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    结果可视化。
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def test_params_flop(model, x_shape):
    total_params = 0
    for param in model.trainable_params():
        param_shape = param.shape
        param_num = np.prod(param_shape)
        total_params += param_num
    print('Number of parameters: {:.2f}M'.format(total_params / 1e6))
    print('Computational complexity (FLOPs) calculation is not implemented for MindSpore.')

class LearningRateScheduler(Callback):

    def __init__(self, optimizer, args):
        super(LearningRateScheduler, self).__init__()
        self.optimizer = optimizer
        #self.scheduler = scheduler
        self.args = args
        self.epoch = 0

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        self.epoch += 1
        adjust_learning_rate(self.optimizer, self.epoch, self.args)


