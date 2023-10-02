import torch
import math
import psutil
import gc
import sys
import os

def weightinit(net):
    weight_stdev = 1
    classname = net.__class__.__name__
    if classname.find('Linear') != -1:
        y = weight_stdev / math.sqrt(net.in_features)
        net.weight.data.uniform_(-y,y)
        if net.bias is not None:
            net.bias.data.uniform_(-y,y)

def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:        
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
    
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std

#https://github.com/facebookresearch/fairseq/issues/401
#By @smth
def memReport():
    for obj in gc.get_objects():
        print(type(obj),sys.getsizeof(obj))
        #if torch.is_tensor(obj):
        #    print(type(obj), obj.size())

def cpuStats():
        print(sys.version)
        print(psutil.cpu_percent())
        print(psutil.virtual_memory())  # physical memory usage
        pid = os.getpid()
        py = psutil.Process(pid)
        memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB
        print('memory GB:', memoryUse)


