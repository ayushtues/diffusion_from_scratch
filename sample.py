import torch
from dataloader import get_dataloader
from diffusion import Diffusion
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from utils import get_values, print_stats
import torchvision.transforms as T
import torchvision
from torchvision import transforms 
import matplotlib.pyplot as plt
import numpy as np 
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

sqrt_alpha_hat_ts, sqrt_alpha_hat_ts_2, alpha_ts, beta_ts, post_std = get_values(device)
model = Diffusion(sqrt_alpha_hat_ts, sqrt_alpha_hat_ts_2, alpha_ts, beta_ts, post_std, 1, 1)
model.load_state_dict(
    torch.load(
        "D:/diffusion/runs/fashion_trainer_20230331_142757/model_20230331_142757_937"
    )
)

def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        # transforms.Lambda(lambda t: torch.clamp(t, 0, 255)), # why is this messing things up?
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    plt.imshow(reverse_transforms(image))

def show_grid_images(x):
    plt.figure(figsize=(15,15))
    plt.axis('off')
    num_images = len(x)

    for i in range(num_images):
      plt.subplot(1, num_images, 1+i)
      show_tensor_image(x[i].detach().cpu())
    plt.savefig("sample.jpg")
    plt.show()  

model = model.to(device)
model.train(True) # okay why is this causing so much difference
x = model.sample(device)
show_grid_images(x)
x = torch.stack(x)
print_stats(x, "x")
print(x.shape)
torchvision.utils.save_image(x, "sample.png")
