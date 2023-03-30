
import torch
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt


def get_cosine_schedule(num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """

    return betas_for_alpha_bar(
        num_diffusion_timesteps,
        lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
    )


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.from_numpy(np.array(betas)).float()

def get_values(device):
  beta_1 = 1e-4
  beta_T = 0.02

  beta_ts = torch.linspace(beta_1, beta_T, 1000)
#   beta_ts = get_cosine_schedule(1000)
  alpha_ts = 1 - beta_ts
  alpha_hat_ts  = torch.cumprod(alpha_ts, 0)
  alpha_hat_ts_prev = F.pad(alpha_hat_ts[:-1], (1, 0), 'constant',1.0)

  sqrt_alpha_ts = torch.sqrt(alpha_ts)
  sqrt_alpha_hat_ts = torch.sqrt(alpha_hat_ts)
  sqrt_alpha_hat_ts_2 = torch.sqrt(1-alpha_hat_ts)
  post_std = torch.sqrt(((1-alpha_hat_ts_prev)/(1-alpha_hat_ts))*beta_ts)

  # prev_sqrt_alpha_hat_ts = sqrt_alpha_hat_ts[:-1] # sqrt(alpha_hat_t-1)
  # prev_alpha_hat_ts = alpha_hat_ts[:-1] # alpha_hat_t-1
  # curr_alpha_hat_ts = alpha_hat_ts[1:] # alpha_hat_t
  # curr_alpha_ts = alpha_ts[1:] # alpha_t
  # curr_sqrt_alpha_hat_ts = sqrt_alpha_hat_ts[1:] # sqrt(alpha_t)
  # curr_sqrt_alpha_hat_ts_2 = torch.sqrt(1-curr_alpha_hat_ts) # sqrt(1 - alpha_hat)
  # curr_beta_ts = beta_ts[1:] # beta_t


  # coeff1 = prev_sqrt_alpha_hat_ts / (1 - curr_alpha_hat_ts)  
  # coeff2 =  (( 1- prev_alpha_hat_ts ) / ( 1- curr_alpha_hat_ts )) * (curr_sqrt_alpha_ts)

  # beta_hat_ts = ((1 - prev_alpha_hat_ts) / (1 - curr_alpha_hat_ts)) * curr_beta_ts
  # coeff_3 = (1-curr_alpha_hat_ts)/curr_sqrt_alpha_hat_ts_2
  sqrt_alpha_hat_ts = sqrt_alpha_hat_ts.to(device)
  sqrt_alpha_hat_ts_2 = sqrt_alpha_hat_ts_2.to(device)
  alpha_ts = alpha_ts.to(device)
  beta_ts = beta_ts.to(device)
  post_std = post_std.to(device)

  return sqrt_alpha_hat_ts, sqrt_alpha_hat_ts_2, alpha_ts, beta_ts, post_std

def print_stats(x, name):
  print(f"{name} max: {torch.max(x)}, min: {torch.min(x)}, mean: {torch.mean(x)}, std: {torch.std(x)}")
