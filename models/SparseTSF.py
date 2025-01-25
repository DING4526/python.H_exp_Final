import mindspore.nn as nn
import mindspore as ms
import mindspore.ops as ops
from mindspore import Tensor
import numpy as np
import mindspore.context as context
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
context.set_context(enable_graph_kernel=True)

class TimeEncoding:
    def __init__(self, period_len, d_model):
        self.period_len = period_len  # 周期长度，例如24小时
        self.d_model = d_model  # 编码的维度，例如128

    def generate(self):
        position = np.arange(self.period_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, self.d_model, 2) * (-np.log(10000.0) / self.d_model))
        encoding = np.zeros((self.period_len, self.d_model))

        # 偶数维度赋值
        encoding[:, 0::2] = np.sin(position * div_term)

        # 奇数维度赋值，补齐奇数维度
        if self.d_model % 2 == 1:
            odd_dim = np.cos(position * div_term)[:, :encoding[:, 1::2].shape[1]]  # 截取到形状匹配
            encoding[:, 1::2] = odd_dim
        else:
            encoding[:, 1::2] = np.cos(position * div_term)

        return encoding

class SP(nn.Cell):
    def __init__(self, configs):
        super(SP, self).__init__()

        # 获取参数
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = configs.period_len
        self.d_model = configs.d_model
        self.model_type = configs.model_type
        self.dropout_rate=configs.dropout_rate
        assert self.model_type in ['linear', 'mlp']

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

        self.conv1d = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=1 + 2 * (self.period_len // 2),
            stride=1,
            padding=self.period_len // 2,
            pad_mode="pad",
            has_bias=False
        )

        self.relu = nn.ReLU()  # 添加激活函数
        self.layer_norm = nn.LayerNorm([self.enc_in]) # 初始化 LayerNorm

        if self.model_type == 'linear':
            self.linear = nn.Dense(self.seg_num_x, self.seg_num_y, has_bias=False)
        elif self.model_type == 'mlp':
            self.mlp1 = nn.SequentialCell([
                nn.Dense(self.seg_num_x, self.d_model),
                nn.ReLU(),
            ])
            self.mlp2=nn.SequentialCell([
                nn.Dropout(p=self.dropout_rate),
                nn.Dense(self.d_model, self.d_model),
            ])
            self.mlp3 = nn.SequentialCell([
                nn.ReLU(),
                nn.Dense(self.d_model, self.seg_num_y)
            ])
            self.mlp=nn.SequentialCell([
                nn.Dense(self.seg_num_x, self.d_model),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_rate),
                nn.Dense(self.d_model, self.seg_num_y)
            ])

        self.bias_conv=nn.Conv2d(
            in_channels=1,out_channels=1,
            kernel_size=(self.seg_num_y + 1, 1),
            stride=1,pad_mode='valid'
        )
        self.bias_mlp = nn.SequentialCell([
            nn.Dense(self.seg_num_x, self.d_model),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Dense(self.d_model, self.seg_num_y)
        ])

        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()
        self.reduce_mean = ops.ReduceMean(keep_dims=True)

    def construct(self, x):  # (batch_size, seq_len, enc_in)
        batch_size = x.shape[0]

        # time_encoding = TimeEncoding(period_len=24, d_model=self.enc_in).generate()# (24, features)
        # time_encoding = np.tile(time_encoding, (self.seq_len // self.period_len, 1))  # 扩展到 seq_len
        # time_encoding = np.tile(time_encoding, (batch_size, 1, 1))  # 扩展到 batch_size
        # time_encoding = Tensor(time_encoding,ms.float32)
        # x = x + time_encoding  # 输入数据与时间编码相加

        # 归一化和转置     b,s,c -> b,c,s
        # x = self.layer_norm(x)  # 层归一化
        seq_mean = self.reduce_mean(x, 1)  # (batch_size, 1, enc_in)
        x = (x - seq_mean)
        x = self.transpose(x, (0, 2, 1))  # (batch_size, enc_in, seq_len)

        # 1D 卷积聚合
        x_reshaped = self.reshape(x, (-1, 1, self.seq_len))  # (batch_size * enc_in, 1, seq_len)
        x_conv = self.conv1d(x_reshaped)  # (batch_size * enc_in, 1, seq_len)
        x_conv = self.reshape(x_conv, (-1, self.enc_in, self.seq_len))  # (batch_size, enc_in, seq_len)
        x_conv=self.relu(x_conv)
        x = x_conv + x  # (batch_size, enc_in, seq_len)

        # 下采样: b,c,s -> bc,n,w -> bc,w,n
        inputs = self.reshape(x, (-1, self.seg_num_x, self.period_len))  # (batch_size * enc_in, seg_num_x, period_len)
        inputs = self.transpose(inputs, (0, 2, 1))  # (batch_size * enc_in, period_len, seg_num_x)


        time_encoding=np.arange(self.seg_num_x)
        time_encoding = (time_encoding - np.mean(time_encoding)) / np.std(time_encoding)
        time_encoding=np.tile(time_encoding,(batch_size*self.enc_in,self.period_len,1))
        time_encoding=Tensor(time_encoding,ms.float32)
        inputs = inputs + time_encoding

        # 稀疏预测
        if self.model_type == 'linear':
            y = self.linear(inputs)  # (batch_size * enc_in, period_len, seg_num_y)
        elif self.model_type == 'mlp':
            y = self.mlp(inputs)  # (batch_size * enc_in, period_len, seg_num_y)

            # # 复杂结构
            # y = self.mlp1(inputs) # (batch_size * enc_in, period_len, d_model)
            # h = self.mlp2(y) # (batch_size * enc_in, period_len, d_model)
            # y = y + h # (batch_size * enc_in, period_len, d_model)
            # y = self.mlp3(y) # (batch_size * enc_in, period_len, seg_num_y)

        # # 偏差块实现
        # concat_op = ops.Concat(axis=2)
        # bias_inputs = concat_op((inputs, y))
        # # (batch_size * enc_in, period_len, seg_num_x + seg_num_y)
        # bias_inputs = self.transpose(bias_inputs,(0, 2, 1))
        # # (batch_size * enc_in, seg_num_x + seg_num_y, period_len)
        # bias_inputs = bias_inputs.unsqueeze(1)
        # # (batch_size * enc_in, 1, seg_num_x + seg_num_y, period_len)
        # bias_inputs = self.bias_conv(bias_inputs)
        # # (batch_size * enc_in, 1, seg_num_x, period_len)
        # bias_inputs = bias_inputs.squeeze(1)
        # # (batch_size * enc_in, seg_num_x, period_len)
        # bias_inputs = self.transpose(bias_inputs, (0, 2, 1))
        # # (batch_size * enc_in, period_len, seg_num_x)
        # # bias_inputs = inputs + bias_inputs 残差连接效果不好，故抛弃
        # y = self.bias_mlp(bias_inputs)
        # # (batch_size * enc_in, period_len, seg_num_y)


        # if self.model_type == 'linear':
        #     y = self.linear(y)  # (batch_size * enc_in, period_len, seg_num_y)
        # elif self.model_type == 'mlp':
        #     y = self.mlp(y)  # (batch_size * enc_in, period_len, seg_num_y)

        # 上采样: bc,w,m -> bc,m,w -> b,c,s
        y = self.transpose(y, (0, 2, 1))  # (batch_size * enc_in, seg_num_y, period_len)
        y = self.reshape(y, (batch_size, self.enc_in, self.pred_len))  # (batch_size, enc_in, pred_len)

        # 转置和去归一化
        y = self.transpose(y, (0, 2, 1)) + seq_mean  # (batch_size, pred_len, enc_in)

        return y
