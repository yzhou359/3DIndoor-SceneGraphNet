import torch
import torch.nn as nn
import torch.nn.init as init
from torch._six import string_classes, int_classes
import re, os
import collections
import numpy as np


def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

_use_shared_memory = False
numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}
def default_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if isinstance(batch[0], list):
        return batch
    elif isinstance(batch[0], torch.Tensor):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))

def try_mkdir(dir):
    try:
        os.makedirs(dir)
    except OSError:
        pass

def get_offset_vec(c1, c2):
    c1_trans_x = (c1['self_info']['translation'][0] - c1['self_info']['dim'][0] * 0.5,
                  c1['self_info']['translation'][0] + c1['self_info']['dim'][0] * 0.5)
    c2_trans_x = (c2['self_info']['translation'][0] - c2['self_info']['dim'][0] * 0.5,
                  c2['self_info']['translation'][0] + c2['self_info']['dim'][0] * 0.5)
    offset_x = np.array([c2_trans_x[0] - c1_trans_x[0], c2_trans_x[0] - c1_trans_x[1],
                         c2_trans_x[1] - c1_trans_x[0], c2_trans_x[1] - c1_trans_x[1]])
    offset_x_idx = int(np.argmin(np.abs(offset_x)))
    offset_x_val = offset_x[offset_x_idx]

    c1_trans_y = (c1['self_info']['translation'][2] - c1['self_info']['dim'][2] * 0.5,
                  c1['self_info']['translation'][2] + c1['self_info']['dim'][2] * 0.5)
    c2_trans_y = (c2['self_info']['translation'][2] - c2['self_info']['dim'][2] * 0.5,
                  c2['self_info']['translation'][2] + c2['self_info']['dim'][2] * 0.5)
    offset_y = np.array([c2_trans_y[0] - c1_trans_y[0], c2_trans_y[0] - c1_trans_y[1],
                         c2_trans_y[1] - c1_trans_y[0], c2_trans_y[1] - c1_trans_y[1]])
    offset_y_idx = int(np.argmin(np.abs(offset_y)))
    offset_y_val = offset_y[offset_y_idx]
    offset = [offset_x_val, offset_y_val]

    # dis = np.sqrt((offset[0]**2 + offset[1]**2))
    delta_x = c1['self_info']['translation'][0] - c2['self_info']['translation'][0]
    delta_y = c1['self_info']['translation'][2] - c2['self_info']['translation'][2]
    dis = np.sqrt(delta_x ** 2 + delta_y ** 2)

    return offset_x_val, offset_y_val, dis

