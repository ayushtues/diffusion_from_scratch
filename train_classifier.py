import torch
from torchvision import transforms 
import matplotlib.pyplot as plt
import numpy as np 
from dataloader import get_dataloader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from utils import get_values, print_stats
from models import Classifier
import torch.nn as nn
import torch.optim as optim
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

dataloader = get_dataloader()

classifier = Classifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters())


def train_one_epoch(epoch_index, batches, tb_writer, run_path, save_freq=500):
    running_loss = 0.0
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(dataloader):
        batch = epoch_index * len(dataloader) + i + 1
        if batch == batches:
            return running_loss / (i + 1)
        x, y, t = data
        y_one = torch.nn.functional.one_hot(y, 10).float()
        x = x.to(device)
        y_one = y_one.to(device)
        # x = x.view(x.shape[0], -1, 1, 1)
        x = x * 2 - 1
        outputs = classifier(x)
        loss = criterion(outputs, y)
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Compute the loss and its gradients
        loss.backward()

        # Adjust learning weights
        optimizer.step()
        loss = loss.detach().cpu().numpy()

        # Gather data and report
        running_loss += loss.item()
        if i % 10 == 0:
            print("  batch {} loss: {}".format(batch, loss))
            tb_writer.add_scalar("Loss/train", loss, batch)
        
        if i % 500 == 0 :
            testloader = get_dataloader(train=False)
            correct = 0
            total = 0
            # since we're not training, we don't need to calculate the gradients for our outputs
            with torch.no_grad():
                for data in testloader:
                    images, labels, _ = data
                    # calculate outputs by running images through the network
                    outputs = classifier(images)
                    # the class with the highest energy is what we choose as prediction
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

        # # Track best performance, and save the model's state
        if i % save_freq == 0:
            model_path = run_path + "/classifier_{}_{}".format(timestamp, batch)
            torch.save(classifier.state_dict(), model_path)

    return running_loss / len(dataloader)


# Initializing in a separate cell so we can easily add more epochs to the same run

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_path = "runs/mnist_classifier{}".format(timestamp)
writer = SummaryWriter(run_path)
epoch_number = 0
classifier = classifier.to(device)
batches = 1000
EPOCHS = int(batches / len(dataloader) + 1)


for epoch in range(EPOCHS):
    print("EPOCH {}:".format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    classifier.train(True)
    avg_loss = train_one_epoch(epoch_number, batches, writer, run_path)
    print(f"EPOCH : {epoch+1} loss : {avg_loss}")
    epoch_number += 1


# model.eval()
# x  = model.sample()
# print("sample:", x.shape)
