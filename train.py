import torch
from dataloader import get_dataloader
from diffusion import Diffusion
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from utils import get_values, print_stats
from models import get_position_embeddings

if torch.cuda.is_available(): 
 dev = "cuda:0" 
else: 
 dev = "cpu" 
device = torch.device(dev) 

dataloader = get_dataloader()
sqrt_alpha_hat_ts, sqrt_alpha_hat_ts_2, alpha_ts, beta_ts = get_values(device)
model = Diffusion(sqrt_alpha_hat_ts, sqrt_alpha_hat_ts_2, alpha_ts, beta_ts, 1, 1)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())


def train_one_epoch(epoch_index, batches, tb_writer, run_path, save_freq=200):
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
        x = x*2 - 1
        t = t.to(device)
        t = t.squeeze(-1)
        t_embed = get_position_embeddings(t, device)
        eps = torch.randn_like(x)


        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        eps_pred = model(x, t, t_embed, eps)

        print_stats(eps, "eps")
        print_stats(eps_pred, "eps_pred")


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

        # # Track best performance, and save the model's state
        if i%save_freq == 0:
            model_path = run_path+'/model_{}_{}'.format(timestamp, batch)
            torch.save(model.state_dict(), model_path)
          
    

    return running_loss/len(dataloader)

# Initializing in a separate cell so we can easily add more epochs to the same run

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
run_path = 'runs/fashion_trainer_{}'.format(timestamp)
writer = SummaryWriter(run_path)
epoch_number = 0
model = model.to(device)
batches = 1000
EPOCHS = int(batches/len(dataloader)+1)


for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, batches, writer, run_path)
    print(f"EPOCH : {epoch+1} loss : {avg_loss}")
    epoch_number += 1
    


# model.eval()
# x  = model.sample()
# print("sample:", x.shape)


