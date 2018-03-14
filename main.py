import numpy as np
import torch
from DataLoader import *

from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as s

load_w = 600
load_h = 450
fine_w = 400
fine_h = 300
data_mean = np.asarray([0.])
batch_size =100

# Construct dataloader
opt_data_train = {
    'data_root': './SUNRGBD/',   # MODIFY PATH ACCORDINGLY
    'load_w': load_w,
    'load_h': load_h,
    'fine_w': fine_w,
    'fine_h': fine_h,
    'data_mean': data_mean,
    'randomize': True
    }


loader_train = DataLoaderDisk(**opt_data_train)

data = loader_train.next_batch(batch_size)

print(np.shape(data[0]),np.shape(data[1]))