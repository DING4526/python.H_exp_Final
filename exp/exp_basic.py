import os
from mindspore import context


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model()

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device_target = "GPU"
            print('Use GPU: {}'.format(self.args.gpu))
        else:
            device_target = "CPU"
            print('Use CPU')
        context.set_context(mode=context.GRAPH_MODE, device_target=device_target)
        if self.args.use_multi_gpu and self.args.use_gpu:
            devices = self.args.devices.replace(' ', '').split(',')
            device_num = len(devices)
            context.set_auto_parallel_context(parallel_mode=context.ParallelMode.DATA_PARALLEL, device_num=device_num)
            print('Using multi-GPU: {}'.format(devices))
        return device_target

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
