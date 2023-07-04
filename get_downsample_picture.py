from downsample import downsample
from utils import load_img
import torch
import torchvision
from PIL import Image

device=torch.device('cuda')
img = load_img('pictures/NOISY.PNG').to(device)
img1, _ = downsample(img)
origin = load_img("pictures/ORIGIN.PNG").to(device)
img2, _ = downsample(origin)
trans = torchvision.transforms.ToPILImage()
pic1 = trans(img1)
pic2 = trans(img2)
pic1.save("pictures/noisy.png")
pic2.save("pictures/origin.png")