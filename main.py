import torch
from dataloader import get_dataloader

from unet import UNet

dataloader = get_dataloader()
x,y,t = next(iter(dataloader))

unet = UNet(1, 1)

t  = torch.rand(64, 32)

print(unet(x, t).shape)






