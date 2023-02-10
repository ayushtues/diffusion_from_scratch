import torch
from dataloader import get_dataloader
from diffusion import Diffusion
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

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


def train_one_epoch(epoch_index, batches, tb_writer):
    running_loss = 0.
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(dataloader):
        batch = epoch_index * len(dataloader) + i + 1
        if(batch==batches):
          return running_loss/(i+1)
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
        if i % 10 == 0:
            print('  batch {} loss: {}'.format(batch, loss))
            tb_writer.add_scalar('Loss/train', loss, batch)
      
    

    return running_loss/len(dataloader)

# Initializing in a separate cell so we can easily add more epochs to the same run

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0
model = model.to(device)
batches = 1000
EPOCHS = int(batches/len(dataloader)+1)


for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, batches, writer)
    print(f"EPOCH : {epoch+1} loss : {avg_loss}")
    epoch_number += 1
    


# model.eval()
# x  = model.sample()
# print("sample:", x.shape)


