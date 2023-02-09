from torch import nn
import torch
from unet import UNet

class Diffusion_forward(nn.Module):
    def __init__(self, curr_sqrt_alpha_ts, curr_sqrt_alpha_hat_ts_2, n_channels, n_classes, bilinear=False):
        super(Diffusion_forward, self).__init__()
        self.unet = UNet(n_channels, n_classes, bilinear=False)
        self.curr_sqrt_alpha_ts = curr_sqrt_alpha_ts
        self.curr_sqrt_alpha_hat_ts_2 = curr_sqrt_alpha_hat_ts_2
    
    def forward(self, x, t, eps):
      t = t.squeeze(-1)
      t_embed = torch.randn(64, 32)

      c1 = torch.gather(self.curr_sqrt_alpha_ts, 0, t).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # TODO, move this to the dataset itself instead of using gather
      c2 = torch.gather(self.curr_sqrt_alpha_hat_ts_2, 0, t).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

      input = x*c1 + eps*c2

      eps_pred = self.unet(input, t_embed)

      return eps_pred

