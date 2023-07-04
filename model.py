import torch
import torch.nn as nn
from torch.nn import functional as F
from downsample import downsample

# class Args():
#     def __init__(self, lr, batch_size=1, num_epoches=1):
#         self.lr = lr
#         self.batch_size = batch_size
#         self.num_epoches = num_epoches

#     def getall(self):
#         return self.lr, self.batch_size, self.num_epoches


class Lightnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, (3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, (3, 3), stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 3, (1, 1))

    def forward(self, X):
        return X + self.conv3(F.relu(self.conv2(F.relu(self.conv1(X)))))

class Loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, src_down1, src_down2, dst, down1_dst, down2_dst):
        loss_res = 0.5 * ((down1_dst - src_down2) ** 2 + (down2_dst - src_down1) ** 2)
        dst_down1, dst_down2 = downsample(dst)
        loss_cons = 0.5 * ((down1_dst - dst_down1) ** 2 + (down2_dst - dst_down2) ** 2)
        return (loss_res + loss_cons).mean()
    

