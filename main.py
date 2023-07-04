from train import train, predict
from utils import load_img, net_init, set_seed
from model import Lightnet
import torch

set_seed(42)
net = Lightnet()
img = load_img('pictures/noisy.png')
net.apply(net_init)
lr = 0.001
num_epochs = 2000
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

train(net, lr, num_epochs, device, img)
predict(net, img, device)
