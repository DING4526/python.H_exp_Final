from mindspore.dataset import GeneratorDataset
import warnings

warnings.filterwarnings('ignore')

from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, Dataset_Solar,Dataset_PRSA,Dataset_AQ

def data_provider(args, flag):
    data_dict = {  # 引入各种自定义的数据集类（保持类名不变）
        'ETTh1': Dataset_ETT_hour,
        'ETTh2': Dataset_ETT_hour,
        'ETTm1': Dataset_ETT_minute,
        'ETTm2': Dataset_ETT_minute,
        'custom': Dataset_Custom,
        'Solar': Dataset_Solar,
        'PRSA':Dataset_PRSA,
        'AQ':Dataset_AQ
    }

    Data = data_dict[args.data]  # Data被赋值为特定的类（Data是一个类）
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False  # 是否打乱数据
        drop_last = False  # 是否丢弃最后一个不完整批次
        batch_size = args.batch_size  # 批次大小
        freq = args.freq  # 数据频率
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq
    )

    # print(data_set)
    print(flag, len(data_set))

    # 使用 MindSpore 的 GeneratorDataset
    loader = GeneratorDataset(
        data_set.generator(),
        column_names=["seq_x", "seq_y", "seq_x_mark", "seq_y_mark"],
        shuffle=shuffle_flag
    )

    loader = loader.batch(
        batch_size,
        drop_remainder=drop_last,
        num_parallel_workers=args.num_workers
    )

    return data_set, loader
