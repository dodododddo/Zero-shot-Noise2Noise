import torch
import torch.nn as nn
from model import Loss
from downsample import downsample
import matplotlib.pyplot as plt
from d2l import torch as d2l
import torchvision
import cv2 as cv
from utils import Timer
from PIL import Image

def train(net, lr, num_epochs, device, img):
    net = net.to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    loss = Loss()
    timer = Timer()
    img = img.to(device)
    down1, down2 = downsample(img)
    down1.to(device)
    down2.to(device)
    loss_list = []
    epoch_list = []

    for epoch in range(num_epochs):
        timer.start()
        optimizer.zero_grad()
        down1_dst = net(down1)
        down2_dst = net(down2)
        dst = net(img)
        l = loss(down1, down2, dst, down1_dst, down2_dst)
        l.backward()
        optimizer.step()
        loss_list.append(l.detach().to(torch.device('cpu')))
        epoch_list.append(epoch + 1)
        timer.stop()
    
    with open('train_datas.log','a+') as f:
        f.write(f'loss {loss_list[-1]:.3f}\n')
        f.write(f'{num_epochs / timer.sum():.1f} examples/sec '
            f'on {str(device)}\n')
    
    plt.plot(epoch_list, loss_list, label=u"loss")
    plt.show()

def predict(net, img, device):
    img = img.to(device)
    res = net(img)
    trans = torchvision.transforms.ToPILImage()
    image = trans(res)
    image.save('pictures/result.jpg')
    image.show()



        
