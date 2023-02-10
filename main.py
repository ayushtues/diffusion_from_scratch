import torch
from dataloader import get_dataloader
from diffusion import Diffusion

if torch.cuda.is_available(): 
 dev = "cuda:0" 
else: 
 dev = "cpu" 
device = torch.device(dev) 

dataloader = get_dataloader()


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

model = Diffusion(curr_sqrt_alpha_ts, curr_sqrt_alpha_hat_ts_2, curr_alpha_ts, curr_beta_ts, 1, 1)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())


def train_one_epoch(epoch_index):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(dataloader):
        print(i)
        x, y, t = data
        x = x.to(device)
        t = t.to(device)
        t_embed = torch.randn(64, 32, device=device)
        eps = torch.randn_like(x)


        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        eps_pred = model(x, t, t_embed, eps)

        # Compute the loss and its gradients
        loss = loss_fn(eps_pred, eps)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        loss = loss.detach().cpu().numpy()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(dataloader) + i + 1
            running_loss = 0.

    return last_loss

# Initializing in a separate cell so we can easily add more epochs to the same run

epoch_number = 0
model = model.to(device)
EPOCHS = 5


for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number)
    print(avg_loss)
    epoch_number += 1
    


model.eval()
x  = model.sample()
print("sample:", x.shape)


