import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import load_img

def downsample(img_tensor:torch.Tensor):
    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    kernel1 = torch.tensor([[[[0.5, 0],[0, 0.5]]], [[[0.5, 0],[0, 0.5]]], [[[0.5, 0],[0, 0.5]]]],requires_grad=False, device=device)
    kernel2 = torch.tensor([[[[0, 0.5], [0.5, 0]]], [[[0, 0.5], [0.5, 0]]], [[[0, 0.5], [0.5, 0]]]], requires_grad=False, device=device)

    return F.conv2d(img_tensor, kernel1, stride=2, groups=3), F.conv2d(img_tensor, kernel2, stride=2, groups=3)

def shape_test(tensor):
    res = downsample(tensor)
    return res[0].shape, res[1].shape

if __name__ == "__main__":
    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    X = torch.ones((1, 3, 64, 48)).to(device)
    print(shape_test(X))
    img = load_img('NOISY.PNG').to(device)
    print(img)
    down_img1, down_img2 = downsample(img)
    print(down_img1.shape)

