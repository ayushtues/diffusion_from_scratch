import torch
from torch import nn
from models import UNet
from models import get_position_embeddings
from utils import print_stats
import torch.nn.functional as F
from copy import deepcopy
class Diffusion(nn.Module):
    def __init__(
        self,
        sqrt_alpha_hat_ts,
        sqrt_alpha_hat_ts_2,
        alpha_ts,
        beta_ts,
        post_std,
        n_channels,
        n_classes,
        bilinear=False,
    ):
        super(Diffusion, self).__init__()
        self.model = UNet(n_channels, n_classes, bilinear=False)
        self.ema_model = deepcopy(self.model)
        self.ema_decay = 0.999
        self.ema_start = 1000
        self.ema_update_rate = 1
        self.step = 0
        # self.model = SimpleUnet()
        # self.model = SimpleMLP()
        # self.model = SimpleMLP2()
        self.sqrt_alpha_hat_ts = sqrt_alpha_hat_ts
        self.sqrt_alpha_hat_ts_2 = sqrt_alpha_hat_ts_2
        self.alpha_ts = alpha_ts
        self.sqrt_alpha_ts = torch.sqrt(alpha_ts)
        self.beta_ts = beta_ts
        self.sigma_ts = torch.sqrt(beta_ts)
        self.alpha_ts_2 = 1 - self.alpha_ts
        self.post_std = post_std

    def update_ema(self):
        self.step += 1
        if self.step % self.ema_update_rate == 0:
            if self.step < self.ema_start:
                self.ema_model.load_state_dict(self.model.state_dict())
            else:
                for current_params, ema_params in zip(self.model.parameters(), self.ema_model.parameters()):
                    old, new = ema_params.data, current_params.data
                    if old is not None:
                        ema_params.data = old * self.ema_decay + new * (1 - self.ema_decay)

    def forward(self, x, t, t_embed, eps, y=None):
        c1 = (
            torch.gather(self.sqrt_alpha_hat_ts, 0, t)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )  # TODO, move this to the dataset itself instead of using gather
        c2 = (
            torch.gather(self.sqrt_alpha_hat_ts_2, 0, t)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )

        input_x = x * c1 + eps * c2

        eps_pred = self.model(input_x, t_embed, y)

        return eps_pred

    @torch.no_grad()
    def sample(self, device, y=None):
        if y is None:
            y = torch.zeros([1], device=device, dtype=torch.long)
            y = F.one_hot(y, 10).float()
        x = torch.randn([1, 1, 32, 32], device=device)
        x_returned = []
        for i in reversed(range(1000)):
            t = get_position_embeddings(i, device).unsqueeze(0)
            eps_pred = self.ema_model(x, t, y)
            eps_pred = (
                self.alpha_ts_2[i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                / self.sqrt_alpha_hat_ts_2[i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            ) * eps_pred
            x = x - eps_pred
            x = x * (
                1 / self.sqrt_alpha_ts[i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            )
            if i != 0:
                z = torch.randn_like(x, device=device)
                z = self.sigma_ts[i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * z
            else:
                z = torch.zeros_like(x, device=device)
            x = x + z

            if i % 50 == 0:
                x_img = (x + 1.0) / 2
                x_returned.append(x_img.squeeze(0).detach())

        return x_returned
