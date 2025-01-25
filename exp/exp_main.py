from fontTools.misc.timeTools import epoch_diff

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import SparseTSF
from utils.tools import EarlyStopping, visual, test_params_flop
from utils.metrics import *

import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Model, Tensor
from mindspore.nn import loss as ms_loss
from mindspore import context, dtype as mstype
import matplotlib.pyplot as plt

import os
import time

import warnings

warnings.filterwarnings('ignore')


class Mixed_Loss(nn.Cell):
    def __init__(self, weight=1.0):
        super(Mixed_Loss, self).__init__()
        self.weight = weight
        self.square = ops.Square()
        self.mean = ops.ReduceMean()
        self.abs = ops.Abs()

    def MSEloss(self, pred, true):
        """均方误差 (MSE) 实现"""
        return self.mean(self.square(pred - true))

    def MAEloss(self, pred, true):
        return self.mean(self.abs(pred - true))

    def RSEloss(self, pred, true):
        """归一化平方误差 (RSE) 实现"""
        numerator = self.mean(self.square(pred - true))
        denominator = self.mean(self.square(true - ops.ReduceMean()(true)))
        return numerator / denominator

    def construct(self, pred, true):
        # 计算损失
        mse_loss = self.MSEloss(pred, true)
        rse_loss = self.RSEloss(pred, true)
        mae_loss = self.MAEloss(pred, true)
        # 混合损失
        loss =  0.4 * mse_loss + 0.2 * rse_loss + mae_loss * 0.4
        return self.weight * loss

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.model = self._build_model()

    def _build_model(self):
        model = SparseTSF.SP(self.args)
        return model

    def _get_data(self, flag):#获取训练、验证或测试数据。
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        optimizer = nn.Adam(params=self.model.trainable_params(),
                            learning_rate=self.args.learning_rate,
                            weight_decay = self.args.weight_decay)  # 添加weight_decay参数
        # weight_decay = self.args.weight_decay
        return optimizer
    def _select_criterion(self):#选择损失函数，根据配置选择 MAE、MSE 或 Smooth L1 损失。
        if self.args.loss == "mae":
            criterion = ms_loss.L1Loss()
        elif self.args.loss == "mse":
            criterion = ms_loss.MSELoss()
        elif self.args.loss == "smooth":
            criterion = ms_loss.SmoothL1Loss()
        elif self.args.loss == "mixed":
            criterion = Mixed_Loss(weight=1.0)
        else:
            criterion = ms_loss.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.set_train(False)
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            batch_x = Tensor(batch_x, dtype=mstype.float32)
            batch_y = Tensor(batch_y, dtype=mstype.float32)

            # encoder - decoder
            outputs = self.model(batch_x)

            if self.args.features=='MS':
                outputs = outputs[:, -self.args.pred_len:, 0:1]
                batch_y = batch_y[:, -self.args.pred_len:, 0:1]
            else :
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :]
            # f_dim = -1 if self.args.features == 'MS' else 0
            # outputs = outputs[:, -self.args.pred_len:, f_dim:]
            # batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

            # 确保 shapes 一致
            if outputs.shape != batch_y.shape:
                print(f"[VALI DEBUG] outputs shape: {outputs.shape}, batch_y shape: {batch_y.shape}")
                raise ValueError("Outputs and batch_y shapes do not match during validation.")

            loss = criterion(outputs, batch_y)
            total_loss.append(loss.asnumpy())

        total_loss = np.average(total_loss)
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        start_time=time.time()
        epoch_count=0
        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # 使用 WithLossCell 和 TrainOneStepCell
        with_loss = nn.WithLossCell(self.model, criterion)
        train_step = nn.TrainOneStepCell(with_loss, model_optim)
        train_step.set_train()

        # 初始化用于记录损失的列表
        train_losses = []
        vali_losses = []
        test_losses = []

        for epoch in range(self.args.train_epochs):
            epoch_count += 1
            iter_count = 0
            train_loss = []
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                batch_x = Tensor(batch_x, dtype=mstype.float32)
                batch_y = Tensor(batch_y, dtype=mstype.float32)

                # 裁剪 y 和 output 在训练步骤中
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                # print("sizeof batch_x:", batch_x.shape)
                # print("sizeof batch_y:",batch_y.shape)
                # print("sizeof batch_x_mark:", batch_x_mark.shape)

                # 执行训练单步，计算损失并更新参数
                loss = train_step(batch_x, batch_y)
                train_loss.append(loss.asnumpy())

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            # 记录损失
            train_losses.append(train_loss)
            vali_losses.append(vali_loss)
            test_losses.append(test_loss)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        end_time=time.time()
        self.epoch_time=(end_time-start_time) / epoch_count
        print("total train time: {}".format(self.epoch_time))


        # 保存最佳模型
        best_model_path = os.path.join(path, 'checkpoint.ckpt')
        ms.save_checkpoint(self.model, best_model_path)

        # 绘制损失曲线
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, 'bo-', label='Train Loss')
        plt.plot(epochs, vali_losses, 'ro-', label='Validation Loss')
        plt.plot(epochs, test_losses, 'go-', label='Test Loss')
        plt.title('Training, Validation and Test Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        loss_curve_path = os.path.join(path, 'loss_curve.png')
        plt.savefig(loss_curve_path)
        plt.close()
        print(f"Loss curve saved to {loss_curve_path}")

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        print("test_data",test_data)

        if test:
            print('loading model')
            best_model_path = os.path.join('./test_checkpoints/' + setting, 'checkpoint.ckpt')
            ms.load_checkpoint(best_model_path, self.model)

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.set_train(False)
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            batch_x = Tensor(batch_x, dtype=mstype.float32)
            batch_y = Tensor(batch_y, dtype=mstype.float32)

            # encoder - decoder
            outputs = self.model(batch_x)

            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
            outputs = outputs.asnumpy()
            batch_y = batch_y.asnumpy()

            pred = outputs
            true = batch_y

            preds.append(pred)
            trues.append(true)

            if i % 20 == 0:
                input = batch_x.asnumpy()
                gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                visual(gt, pd, os.path.join(folder_path, setting + str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop(self.model, (batch_x.shape[1], batch_x.shape[2]))
            exit()

        # 拼接所有预测结果和真实值
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        # # 保存结果
        # folder_path = './results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}, time:{}'.format(mse, mae, rse, self.epoch_time))
        with open("result.txt", 'a') as f:
            f.write(setting + "  \n")
            f.write('mse:{}, mae:{}, rse:{}, time:{}'.format(mse, mae, rse, self.epoch_time))
            f.write('\n')
            f.write('\n')

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        print('pred_data:',pred_data)

        if load:
            print('loading model')
            best_model_path = os.path.join('./test_checkpoints/' + setting, 'checkpoint.ckpt')
            ms.load_checkpoint(best_model_path, self.model)

        preds = []

        self.model.set_train(False)
        for i, (batch_x) in enumerate(pred_loader):
            batch_x = Tensor(batch_x, dtype=mstype.float32)

            # encoder - decoder
            outputs = self.model(batch_x)

            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            pred = outputs.asnumpy()
            preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # 保存预测结果
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
