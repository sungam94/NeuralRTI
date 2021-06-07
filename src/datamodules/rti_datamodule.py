from typing import Optional

from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from src.datamodules.datasets.lp_rti_dataset import LpDataset

class LpDataModule(LightningDataModule):
    def __init__(self, batch_size: int = 1, ratio: tuple = (1, 1), **kwargs):
        super().__init__()

        self.batch_size = batch_size
        self.ratio = ratio
        self.setup()

    def setup(self, stage: Optional[str] = None):
        self.data_train = LpDataset(self.hparams, split="train", ratio=self.ratio)
        self.data_val = LpDataset(self.hparams, split="eval", ratio=self.ratio)
        self.data_test = LpDataset(self.hparams, split="test", ratio=self.ratio)

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            shuffle=True,
            batch_size=self.batch_size,
        )

    def val_dataloader(self):
        return DataLoader(self.data_val,
                          pin_memory=True,
                          num_workers=self.hparams.num_workers,
                          batch_size=1)

    def test_dataloader(self):
        return DataLoader(self.data_test,
                          batch_size=1)