import torch
from torch import nn
from unet import UNet

class Diffusion(nn.Module):
    def __init__(self, curr_sqrt_alpha_ts, curr_sqrt_alpha_hat_ts_2, curr_alpha_ts, curr_beta_ts, n_channels, n_classes, bilinear=False):
        super(Diffusion, self).__init__()
        self.unet = UNet(n_channels, n_classes, bilinear=False)
        self.curr_sqrt_alpha_ts = curr_sqrt_alpha_ts
        self.curr_sqrt_alpha_hat_ts_2 = curr_sqrt_alpha_hat_ts_2
        self.curr_alpha_ts = curr_alpha_ts
        self.curr_beta_ts = curr_beta_ts
        self.sigma_ts = torch.sqrt(curr_beta_ts)
        self.curr_alpha_ts_2 = 1 - self.curr_alpha_ts

    
    def forward(self, x, t, eps):
      t = t.squeeze(-1)
      t_embed = torch.randn(64, 32)

      c1 = torch.gather(self.curr_sqrt_alpha_ts, 0, t).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # TODO, move this to the dataset itself instead of using gather
      c2 = torch.gather(self.curr_sqrt_alpha_hat_ts_2, 0, t).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

      input = x*c1 + eps*c2

      eps_pred = self.unet(input, t_embed)

      return eps_pred
    
    def sample(self):
      x = torch.randn([1, 1, 28, 28])
      for i in reversed(range(99)):
        t = torch.rand([1, 32])
        eps_pred = self.unet(x, t)
        eps_pred = (self.curr_alpha_ts_2[i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) / self.curr_sqrt_alpha_hat_ts_2[i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) ) * eps_pred
        x = x - eps_pred
        x = x * (1/self.curr_sqrt_alpha_ts[i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
        if i!=0:
          z = torch.rand_like(x)
          z = self.sigma_ts[i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)*z
        else:
          z = torch.zeros_like(x)
        x = x + z

      return x





