import numpy as np
import os, json
from utils.utl import try_mkdir

''' CHANGE TO YOUR OWN DATASET DIRECTORY '''
root_dir = r'/YOUR/DATASET/DIRECTORY'

pkl_dir = os.path.join(root_dir, 'data')
log_dir = os.path.join(root_dir, 'nn', 'logs')
ckpt_dir = os.path.join(root_dir, 'nn', 'ckpts')

try_mkdir(os.path.join(root_dir, 'nn'))
try_mkdir(log_dir)
try_mkdir(ckpt_dir)


id2type = np.loadtxt('data/preprocess/SUNCG_id2type.csv', delimiter=',', dtype=str)
dic_id2type = {}
dic_detail2causal = {}
for line in id2type:
    dic_id2type[line[1]] = (line[2], line[3])
    dic_detail2causal[line[2]] = line[3]

k_size_dic = {'bedroom':58, 'living':58, 'bathroom':38, 'office':49}
