import torch
from dataloader import get_dataloader

dataloader = get_dataloader()
x,y,t = next(iter(dataloader))

print(t.shape)




