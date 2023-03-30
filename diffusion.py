import torch
from torch import nn
from models import UNet, SimpleMLP, SimpleMLP2, SimpleUnet
from models import get_position_embeddings
from utils import print_stats
import torch.nn.functional as F
from unet import UNet2


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
        # self.model = UNet(n_channels, n_classes, bilinear=False)
        self.model = UNet2(
            img_channels=1,
            base_channels=32,
            channel_mults=(1, 2, 2, 2),
            time_emb_dim=32*4,
            norm="gn",
            dropout=0.1,
            activation=F.silu,
            attention_resolutions=(1,),
            num_classes=None,
            initial_pad=0,
        )
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

    def forward(self, x, t, t_embed, eps):
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

        eps_pred = self.model(input_x, t_embed)

        return eps_pred

    @torch.no_grad()
    def sample(self, device):
        # x = torch.randn([1, 1, 28, 28], device=device)
        x = torch.randn([1, 1, 32, 32], device=device)
        # x = torch.randn([1, 2, 1, 1], device=device)
        x_returned = []
        for i in reversed(range(1000)):
            t = get_position_embeddings(i, device).unsqueeze(0)
            t = torch.ones([1], device=device)*i
            # print("x1:", x.shape)
            eps_pred = self.model(x, t)
            # print_stats(x, f"eps_pred_{i}")
            eps_pred = (
                self.alpha_ts_2[i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                / self.sqrt_alpha_hat_ts_2[i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            ) * eps_pred
            # print_stats(x, f"eps_pred2_{i}")
            x = x - eps_pred
            x = x * (
                1 / self.sqrt_alpha_ts[i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            )
            # print("x2:", x.shape)

            if i != 0:
                z = torch.rand_like(x, device=device)
                z = self.post_std[i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * z
            else:
                z = torch.zeros_like(x, device=device)
            x = x + z

            # print_stats(x, i)

            if i % 50 == 0:
                x_img = (x + 1.0) / 2
                x_returned.append(x_img.squeeze(0).detach())

        return x_returned
