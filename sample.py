import torch
from dataloader import get_dataloader
from diffusion import Diffusion
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from utils import get_values, print_stats
import torchvision.transforms as T
import torchvision

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

sqrt_alpha_hat_ts, sqrt_alpha_hat_ts_2, alpha_ts, beta_ts, post_std = get_values(device)
model = Diffusion(sqrt_alpha_hat_ts, sqrt_alpha_hat_ts_2, alpha_ts, beta_ts, post_std, 1, 1)
model.load_state_dict(
    torch.load(
        "/content/drive/MyDrive/diffusion/diffusion_from_scratch/runs/fashion_trainer_20230306_052927/model_20230306_052927_938"
    )
)

model = model.to(device)
model.eval()
x = model.sample(device)
x = torch.stack(x)
print_stats(x, "x")
print(x.shape)
torchvision.utils.save_image(x, "sample.png")
