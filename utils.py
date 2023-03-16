
import torch
import torch.nn.functional as F


def get_values(device):
  beta_1 = 1e-4
  beta_T = 0.02

  beta_ts = torch.linspace(beta_1, beta_T, 1000)
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
