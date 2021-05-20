import os
import torch
import torch.nn.functional as F
from pytorch_lightning.utilities.cli import LightningCLI
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from argparse import ArgumentParser
from torchvision import transforms as tv

import pytorch_lightning as pl
# from pytorch_lightning.utilities.cli import LightningCLI
# from pytorch_lightning.utilities.imports import _TORCHVISION_AVAILABLE

from PIL import Image
from typing import Optional
import utils

class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument('--data_dir', type=str, default='/home/mk301/RTI/loewenkopf')
        parser.add_argument('--lp_name', type=str, default='dirs.lp')


class LpDataset(Dataset):
    def __init__(self):
        print('init dataset')
        self.data_dir = args.data_dir
        self.lp_filename = args.lp_name
        self.img_wh = None
        self.num_lights = None

        self.all_dirs = []
        self.all_rgb = []
        self.image_paths = []
 
        self.define_transforms()
        self.load_lp_file()

    def load_lp_file(self):
        print("loading lp file")
        with open(self.data_dir + '/' + self.lp_filename) as f:
            data = f.read()
        lines = data.split('\n')
        self.num_lights = int(lines[0])

        for i, l in enumerate(lines):
            s = l.split(" ")
            if len(s) == 4:
                # read image files
                image_path = os.path.join(self.data_dir, s[0])
                self.image_paths += [image_path]
                # read light directions
                dir_T =  torch.FloatTensor([list(map(float, s[1:4]))])
                self.all_dirs += [dir_T]

                # # load image
                # img = Image.open(image_path).convert('RGB')
                # if self.img_wh is not None:
                #     img = img.resize(self.img_wh, Image.LANCZOS)
                # else:
                #     self.img_wh = img.size
                # img = self.transform(img) # (H x W x C) [0, 255] to (C x H x W) [0.0, 1.0]
                # img= img.permute(1, 2, 0) # (H, W, C)
                # self.all_rgb += [img]
        self.all_dirs = torch.cat(self.all_dirs, 0) # num_img*h*w, 3)
        # self.all_rgb = torch.cat(self.all_rgb, -1) # (num_img, 3)

    def __getitem__(self, idx):
        l_dir = self.all_dirs[idx]
        # load image
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.img_wh is not None:
            img = img.resize(self.img_wh, Image.LANCZOS)
        else:
            self.img_wh = img.size
        img = self.transform(img)  # (H x W x C) [0, 255] to (C x H x W) [0.0, 1.0]
        img = img.permute(1, 2, 0)  # (H, W, C)
        return {'dir': l_dir,
                  'img': img}

    def __len__(self):
        assert self.num_lights == len(self.all_dirs)
        return self.num_lights

    def define_transforms(self):
        # Not sure why this is not in __init__
        self.transform = tv.ToTensor()


class LpDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "/home/mk301/RTI/loewenkopf", batch_size: int = 1,):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        dataset = LpDataset()
        self.data_test = LpDataset()
        self.data_train, self.data_val = random_split(dataset, [200, 40])

        # self.mnist_test = MNIST(self.data_dir, train=False)
        # mnist_full = MNIST(self.data_dir, train=True)
        # self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size)


class NeuralRtiModule(pl.LightningModule):
    """
    encoder, decoder, model = rm.relight_model(l * 3, 9)

def relight_model(l1,compcoeff):
    keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    inputs1 = Input((l1,))
    ld = Input(shape=(2,))
    encoded1 = Dense(l1, activation='elu', name='dense0')(inputs1)
    encoded2 = Dense(l1, activation='elu', name='dense1')(encoded1)
    encoded3 = Dense(l1, activation='elu', name='dense11')(encoded2)

    encoded = Dense(compcoeff, activation='elu', name='dense12')(encoded3)

    encoder = Model(inputs1, encoded)
    encoder.compile(optimizer=keras.optimizers.Adadelta(), loss='mean_squared_error')
    encode_input = Input(shape=(compcoeff,))
    xx = concatenate([encode_input, ld], axis=-1)
    x = Dense(l1, activation='elu', name='dense2')(xx)
    x = Dense(l1, activation='elu', name='dense3')(x)
    x = Dense(l1, activation='elu', name='dense4')(x)
    decoded = Dense(3, name='dense8')(x)
    decoder = Model([encode_input, ld], decoded)
    autoencoder = Model([inputs1, ld], decoder([encoder(inputs1), ld]))
    return encoder, decoder, autoencoder


    """

    def __init__(self, n_lights, coeff ):
        super().__init__()
        # the first three layers contain 3N units, N are number of input lights
        n_units = 3 * n_lights
        self.encoder = nn.Sequential(
            nn.Linear(n_units, n_units),
            nn.ELU(),
            nn.Linear(n_units, n_units),
            nn.ELU(),
            nn.Linear(n_units, n_units),
            nn.ELU(),
            nn.Linear(n_units, coeff)
        )
        self.decoder = nn.Sequential(
            nn.Linear(coeff+2, n_units),
            nn.ELU(),
            nn.Linear(n_units, n_units),
            nn.ELU(),
            nn.Linear(n_units, n_units),
            nn.ELU(),
            nn.Linear(n_units, 3)
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



def cli_main(args):
    # net_module = NeuralRtiModule()

    cli = MyLightningCLI(NeuralRtiModule, LpDataModule, seed_everything_default=1234)
    # cli.trainer.test(cli.model, datamodule=cli.datamodule)


if __name__ == '__main__':
    import sys
    print(sys.argv)

    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--data_dir', type=str, default='/home/mk301/RTI/loewenkopf')
    parser.add_argument('--lp_name', type=str, default='dirs.lp')

    # parser = pl.Trainer.add_argparse_args(parser)
    # parser = LitClassifier.add_model_specific_args(parser)
    args = parser.parse_args()

    cli_main(args)



