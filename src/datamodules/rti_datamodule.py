from typing import Optional

from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from src.datamodules.datasets.lp_rti_dataset import LpDataset


class LpDataModule(LightningDataModule):
    def __init__(self, batch_size: int = 1, train_val_test_split=None, **kwargs):
        super().__init__()
        self.data_dir = kwargs['data_dir']
        self.lp_name = kwargs['lp_name']
        self.num_workers = kwargs['num_workers']
        self.batch_size = batch_size
        self.train_val_test_split = train_val_test_split
        self.setup()

    def setup(self, stage: Optional[str] = None):
        self.data_train = LpDataset(self.data_dir, self.lp_name, split="train", ratio=self.train_val_test_split)
        self.data_val = LpDataset(self.data_dir, self.lp_name, split="eval", ratio=self.train_val_test_split)
        self.data_test = LpDataset(self.data_dir, self.lp_name, split="test", ratio=self.train_val_test_split)

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            batch_size=self.batch_size,
        )

    def val_dataloader(self):
        return DataLoader(self.data_val,
                          pin_memory=True,
                          num_workers=self.num_workers,
                          batch_size=1)

    def test_dataloader(self):
        return DataLoader(self.data_test,
                          batch_size=1)
