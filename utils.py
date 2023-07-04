import torchvision
import numpy as np
import time
import torch.nn as nn
import torch
import random
from PIL import Image

def set_seed(seed = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

def net_init(net):
    for param in net.parameters():
        if type(param) == nn.Conv2d:
            nn.init.xavier_uniform_(param)

def load_img(image:str):
    trans = torchvision.transforms.ToTensor()
    img = Image.open(image)
    return trans(img)

class Timer(object):
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        self.tik = time.time()
    
    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]
    
    def avg(self):
        return sum(self.times) / len(self.times)
    
    def sum(self):
        return sum(self.times)
    
    def cumsum(self):
        return np.array(self.times).cumsum().tolist