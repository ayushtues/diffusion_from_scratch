
import torch

def get_values(device):
  beta_1 = 1e-4
  beta_T = 0.02

  beta_ts = torch.linspace(beta_1, beta_T, 100)
  alpha_ts = 1 - beta_ts
  alpha_hat_ts  = torch.cumprod(alpha_ts, 0)
  sqrt_alpha_ts = torch.sqrt(alpha_ts)
  sqrt_alpha_hat_ts = torch.sqrt(alpha_hat_ts)

  prev_sqrt_alpha_hat_ts = sqrt_alpha_hat_ts[:-1] # sqrt(alpha_hat_t-1)
  prev_alpha_hat_ts = alpha_hat_ts[:-1] # alpha_hat_t-1
  curr_alpha_hat_ts = alpha_hat_ts[1:] # alpha_hat_t
  curr_alpha_ts = alpha_ts[1:] # alpha_t
  curr_sqrt_alpha_ts = sqrt_alpha_ts[1:] # sqrt(alpha_t)
  curr_sqrt_alpha_hat_ts_2 = torch.sqrt(1-curr_alpha_hat_ts) # sqrt(1 - alpha_hat)
  curr_beta_ts = beta_ts[1:] # beta_t


  coeff1 = prev_sqrt_alpha_hat_ts / (1 - curr_alpha_hat_ts)  
  coeff2 =  (( 1- prev_alpha_hat_ts ) / ( 1- curr_alpha_hat_ts )) * (curr_sqrt_alpha_ts)

  beta_hat_ts = ((1 - prev_alpha_hat_ts) / (1 - curr_alpha_hat_ts)) * curr_beta_ts
  coeff_3 = (1-curr_alpha_hat_ts)/curr_sqrt_alpha_hat_ts_2
  curr_sqrt_alpha_ts = curr_sqrt_alpha_ts.to(device)
  curr_sqrt_alpha_hat_ts_2 = curr_sqrt_alpha_hat_ts_2.to(device)
  curr_alpha_ts = curr_alpha_ts.to(device)
  curr_beta_ts = curr_beta_ts.to(device)

  return curr_sqrt_alpha_ts, curr_sqrt_alpha_hat_ts_2, curr_alpha_ts, curr_beta_ts