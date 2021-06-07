import numpy as np
import torch
from torch.utils.data import Dataset


class LpDataset(Dataset):
    reload = 1
    train_idx = None
    test_idx = None
    img_wh = None
    num_train_img = None
    num_test_img = None
    trn_val_dirs = None
    n_l_dirs = None
    h, w, n, c = None, None, None, None

    def __init__(self, args, split="train", ratio=(1, 1)):
        self.data_dir = args.data_dir
        self.lp_filename = args.lp_name
        self.split = split
        assert split in ['train', 'eval', 'test']
        if LpDataset.reload:
            ration_sum = sum(ratio)
            LpDataset.ratio = int(ratio[0] / ratio[1])
            LpDataset.all_dir, LpDataset.all_rgb, LpDataset.img_wh = utils.load_lp_file(
                self.data_dir,
                img_wh=args.val_size
            )


            LpDataset.n_l_dirs = len(LpDataset.all_dir)
            nl = LpDataset.n_l_dirs
            part_size = nl // ration_sum

            LpDataset.num_test_img = part_size * ratio[1]
            LpDataset.test_idx = np.random.choice(
                nl, LpDataset.num_test_img, replace=False
            )
            LpDataset.test_idx = np.sort(LpDataset.test_idx)
            LpDataset.train_idx = list(set(range(nl)) - set(LpDataset.test_idx))

            LpDataset.num_train_img = len(LpDataset.train_idx)
            LpDataset.trn_val_dirs = torch.stack(
                [LpDataset.all_dir[i] for i in LpDataset.train_idx]
            )
            LpDataset.reload = 0

        if self.split == "train":
            self.train_rgb = [LpDataset.all_rgb[i] for i in LpDataset.train_idx]
            self.train_rgb = torch.stack(self.train_rgb)
            self.t_n, self.t_h, self.t_w, self.t_c = self.train_rgb.size()

            self.train_rgb = self.train_rgb.permute(1, 2, 0, 3).contiguous()

            self.train_rgb = self.train_rgb.view(self.t_h * self.t_w,
                                                    self.t_n,
                                                    self.t_c)

            assert self.train_rgb.size() == (self.t_w * self.t_h, self.t_n, self.t_c)

        elif self.split == "eval":
            self.eval_rgb = torch.stack(
                [LpDataset.all_rgb[i] for i in LpDataset.train_idx]
            )
            n, h, w, c = self.eval_rgb.size()
            # self.eval_rgb = self.eval_rgb.permute(0, 2, 3, 1)
            self.eval_rgb = torch.reshape(self.eval_rgb, (n, h * w, c))

            assert self.trn_val_dirs.size() == (LpDataset.num_train_img, 3)
            assert self.eval_rgb.size() == (
                LpDataset.num_train_img,
                self.img_wh[0] * self.img_wh[1],
                3,
            )

        elif self.split == "test":
            self.test_rgb = torch.stack(
                [LpDataset.all_rgb[i] for i in LpDataset.test_idx]
            )
            n, h, w, c = self.test_rgb.size()
            # self.test_rgb = self.test_rgb.permute(0, 3, 2, 1)
            self.test_rgb = torch.reshape(self.test_rgb, (n, w * h, c))

            self.test_dirs = [LpDataset.all_dir[i] for i in LpDataset.test_idx]
            self.test_dirs = torch.stack(self.test_dirs)

            # picked by light direction
            assert self.test_dirs.size() == (LpDataset.num_test_img, 3)
            assert self.test_rgb.size() == (
                LpDataset.num_test_img,
                self.img_wh[0] * self.img_wh[1],
                3,
            )

    def __getitem__(self, idx):
        if self.split == "train":
            t = idx % (self.t_n)
            p = idx % (self.t_w * self.t_h)
            ray = self.train_rgb[p, :]
            gt = ray[t]
            assert len(ray) == self.num_train_img
            return self.trn_val_dirs[t], ray, gt

        elif self.split == "eval":
            l_idx = torch.randint(0, len(LpDataset.trn_val_dirs), (1,))
            return self.trn_val_dirs[l_idx], self.eval_rgb, l_idx

        elif self.split == "test":
            idx = torch.randint(0, LpDataset.num_test_img, (1,))
            return self.test_dirs[idx], self.test_rgb, idx

    def __len__(self):
        if self.split == "train":
            return LpDataset.img_wh[0] * LpDataset.img_wh[1] * self.num_train_img
        elif self.split == "eval":
            return 1
        elif self.split == "test":
            return 1