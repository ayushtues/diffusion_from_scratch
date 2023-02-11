import torch
from dataloader import get_dataloader
from diffusion import Diffusion
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from utils import get_values
import torchvision.transforms as T
import torchvision

if torch.cuda.is_available(): 
 dev = "cuda:0" 
else: 
 dev = "cpu" 
device = torch.device(dev) 

curr_sqrt_alpha_ts, curr_sqrt_alpha_hat_ts_2, curr_alpha_ts, curr_beta_ts = get_values(device)
model = Diffusion(curr_sqrt_alpha_ts, curr_sqrt_alpha_hat_ts_2, curr_alpha_ts, curr_beta_ts, 1, 1)
model.load_state_dict(torch.load("/content/drive/MyDrive/diffusion/diffusion_from_scratch/runs/fashion_trainer_20230210_123428/model_20230210_123428_938"))

model = model.to(device)
model.eval()
x  = model.sample(device)
print(len(x))
print(x[0].shape)
torchvision.utils.save_image(x, 'sample.png')