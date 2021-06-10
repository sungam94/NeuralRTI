import torch
from torch import nn

class NeuralRtiEncoder(nn.Module):
    def __init__(self, n_coeff, n_lights):
        super().__init__()
        n_units = 3 * n_lights
        self.encoder = nn.Sequential(
            # nn.BatchNorm1d(n_units, momentum=0.01, eps=0.001),
            nn.Linear(n_units, n_units),
            # nn.BatchNorm1d(n_units, momentum=0.01, eps=0.001),
            # nn.Dropout(),
            nn.SELU(),
            nn.Linear(n_units, n_units),
            # nn.Dropout(),
            # nn.BatchNorm1d(n_units, momentum=0.01, eps=0.001),
            nn.SELU(),
            nn.Linear(n_units, n_coeff),
            # nn.BatchNorm1d(n_coeff, momentum=0.01, eps=0.001),
            nn.SELU(),
            # nn.BatchNorm1d(n_coeff, momentum=0.01, eps=0.001),
        )

    def forward(self,  ray):
        embedding = self.encoder(ray)
        return embedding


class NeuralRtiDecoder(nn.Module):
    def __init__(self, n_coeff):
        super().__init__()
        # self.norm = nn.BatchNorm1d(2, momentum=0.01, eps=0.001)
        self.decoder = nn.Sequential(

            nn.Linear(n_coeff + 2, n_coeff + 2),
            nn.SELU(),
            nn.Linear(n_coeff + 2, n_coeff + 2),
            nn.SELU(),
            nn.Linear(n_coeff + 2, n_coeff + 2),
            nn.SELU(),
            nn.Linear(n_coeff + 2, 3),
            # nn.ELU()
        )

    def forward(self, embedding, l_dir):
        # l_dir =self.norm(l_dir)
        latent_code = torch.cat([embedding, l_dir], dim=-1)
        rgb = self.decoder(latent_code)
        return rgb