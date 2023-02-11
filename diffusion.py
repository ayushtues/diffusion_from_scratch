import torch
from torch import nn
from models import UNet, SimpleMLP
from models import get_position_embeddings

class Diffusion(nn.Module):
    def __init__(self, curr_sqrt_alpha_ts, curr_sqrt_alpha_hat_ts_2, curr_alpha_ts, curr_beta_ts, n_channels, n_classes, bilinear=False):
        super(Diffusion, self).__init__()
        # self.model = UNet(n_channels, n_classes, bilinear=False)
        self.model = SimpleMLP()
        self.curr_sqrt_alpha_ts = curr_sqrt_alpha_ts
        self.curr_sqrt_alpha_hat_ts_2 = curr_sqrt_alpha_hat_ts_2
        self.curr_alpha_ts = curr_alpha_ts
        self.curr_beta_ts = curr_beta_ts
        self.sigma_ts = torch.sqrt(curr_beta_ts)
        self.curr_alpha_ts_2 = 1 - self.curr_alpha_ts

    
    def forward(self, x, t, t_embed, eps):
      c1 = torch.gather(self.curr_sqrt_alpha_ts, 0, t).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # TODO, move this to the dataset itself instead of using gather
      c2 = torch.gather(self.curr_sqrt_alpha_hat_ts_2, 0, t).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

      input = x*c1 + eps*c2

      eps_pred = self.model(input, t_embed)

      return eps_pred
    
    def sample(self, device):
      x = torch.randn([1, 1, 28, 28], device=device)
      x_returned = []
      for i in reversed(range(99)):
        t = get_position_embeddings(i, device).unsqueeze(0)
        eps_pred = self.model(x, t)
        eps_pred = (self.curr_alpha_ts_2[i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) / self.curr_sqrt_alpha_hat_ts_2[i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) ) * eps_pred
        x = x - eps_pred
        x = x * (1/self.curr_sqrt_alpha_ts[i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
        if i!=0:
          z = torch.rand_like(x, device=device)
          z = self.sigma_ts[i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)*z
        else:
          z = torch.zeros_like(x, device=device)
        x = x + z

        if i%10 == 0:
          x_returned.append(x.squeeze(0).detach())

      return x_returned





