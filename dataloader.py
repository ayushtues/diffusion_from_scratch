import torch
import torchvision
from PIL import Image
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset


class Mnist_custom(torchvision.datasets.MNIST):
    def __init__(self, **kwrgs):
        super().__init__(**kwrgs)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode="L")
        img = img.resize((32, 32))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        time = torch.randint(
            0, 1000, (1,)
        )  # time value needed in diffusion models, might add time embeddings here only in future

        return img, target, time


class Simple2dDataset(Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = torch.rand(1)
        x = x.repeat(2)
        time = torch.randint(
            0, 1000, (1,)
        )  # time value needed in diffusion models, might add time embeddings here only in future
        return x, 1, time


def get_dataloader():
    mnist = Mnist_custom(root='./data', train=True, download=True, transform=ToTensor())
    # simple2d = Simple2dDataset()
    dataloader = DataLoader(mnist, batch_size=128, shuffle=True, drop_last=True)
    # dataloader = DataLoader(simple2d, batch_size=64, shuffle=True, drop_last=True)
    return dataloader
