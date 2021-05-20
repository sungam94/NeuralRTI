import os
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from torchvision import transforms as tv

import pytorch_lightning as pl
from pl_examples import _DATASETS_PATH, _TORCHVISION_MNIST_AVAILABLE, cli_lightning_logo
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.utilities.imports import _TORCHVISION_AVAILABLE

from PIL import Image
import utils

class LitAutoEncoder(pl.LightningModule):
    """
    >>> LitAutoEncoder()  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    LitAutoEncoder(
      (encoder): ...
      (decoder): ...
    )
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 28 * 28),
        )

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('valid_loss', loss, on_step=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('test_loss', loss, on_step=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class MyDataModule(pl.LightningDataModule):

    def __init__(
        self,
        batch_size: int = 32,
    ):
        super().__init__()
        dataset = MNIST(_DATASETS_PATH, train=True, download=True, transform=transforms.ToTensor())
        self.mnist_test = MNIST(_DATASETS_PATH, train=False, download=True, transform=transforms.ToTensor())
        self.mnist_train, self.mnist_val = random_split(dataset, [55000, 5000])
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)


class LpDataset(Dataset):
    def __init__(self, data_dir="/home/mk301/RTI/loewenkopf", lp_name='dirs.lp'):
        print('init dataset')
        self.data_dir = data_dir
        self.lp_filename = lp_name
        self.img_wh = None

        self.l_dirs = []
        self.im_list = []
        self.image_paths = []
 
        self.define_transforms()
        self.load_lp_file()

    def load_lp_file(self):
        print("loading lp file")
        f = open(self.data_dir + '/' + self.lp_filename)
        data = f.read()
        f.close
        linesn = data.split('\n')
        self.num_lights = int(linesn[0])
        print(f"num lines: {len(linesn)}")
        print(f"num lights: {self.num_lights}")
        lines = linesn[1:]
       #### read light directions

        for l in lines:
            s = l.split(" ")
            if len(s) == 4:
                # read image files
                image_path = os.path.join(self.data_dir, s[0])
                self.image_paths += [image_path]
                img = Image.open(image_path)
                if self.img_wh is not None:
                    img = img.resize(self.img_wh, Image.LANCZOS)
                if self.img_wh is None:
                    self.img_wh = img.size
                img = self.transform(img) # (4, h, w)
                img = img.view(4, -1).permute(1, 0) # (h*w, 4) RGBA
                img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # (h*w, 3) blend A to RGB 
                # print(img.size()) 
                # self.im_list.append(img)
                self.im_list += img
                # read light directions
                self.l_dirs += torch.as_tensor([list(map(float, s[1:4]))])
        # print(self.l_dirs)
        print(self.img_wh)
        print(len(self.im_list) )
        print(len(self.im_list) // (self.img_wh[0]*self.img_wh[1]))

        self.l_dirs = torch.cat(self.l_dirs, 0) # (len(self.meta['frames])*h*w, 3)
        self.im_list = torch.cat(self.im_list, 0) # (len(self.meta['frames])*h*w, 3)
        print (self.im_list.size())

    def __getitem__(self, idx):
        l_dir = self.l_dirs[idx:]
        img = self.im_list[idx:]
        sample = {'dir': l_dir,
                  'img': img}
        return sample

    def __len__(self):
        assert num_lights == len(self.l_dirs)
        return self.num_lights

    def define_transforms(self):
        self.transform = tv.ToTensor()



def perpare_data(self):
    self.train_dataset
    self.val_dataset

def cli_main():
    cli = LightningCLI(LitAutoEncoder, MyDataModule, seed_everything_default=1234)
    cli.trainer.test(cli.model, datamodule=cli.datamodule)


if __name__ == '__main__':
    data_path = "/home/mk301/RTI/loewenkopf"
    ds = LpDataset()
    # print(ds[0])
