import torch
from torch import nn

class NeuralRtiEncoder(nn.Module):
    def __init__(self, n_coeff, n_lights):
        super().__init__()
        n_units = 3 * n_lights
        self.encoder = nn.Sequential(
            # nn.BatchNorm1d(n_units, momentum=0.99, eps=0.001),
            nn.Linear(n_units, n_units),
            # nn.BatchNorm1d(n_units, momentum=0.99, eps=0.001),
            nn.ELU(),
            nn.Linear(n_units, n_units),
            # nn.BatchNorm1d(n_units, momentum=0.99, eps=0.001),
            nn.ELU(),
            nn.Linear(n_units, n_units),
            # nn.BatchNorm1d(n_units, momentum=0.99, eps=0.001),
            nn.ELU(),
            nn.Linear(n_units, n_coeff),
#             # nn.BatchNorm1d(n_coeff, momentum=0.99, eps=0.001),
            nn.ELU(),
        )

    def forward(self,  ray):
        embedding = self.encoder(ray)
        return embedding


class NeuralRtiDecoder(nn.Module):
    def __init__(self, n_coeff):
        super().__init__()
#         # self.norm = nn.BatchNorm1d(2, momentum=0.99, eps=0.001)
        self.decoder = nn.Sequential(
            # nn.BatchNorm1d(n_coeff + 2, momentum=0.99, eps=0.001),
            # nn.Linear(n_coeff + 2, n_coeff + 2),
            # nn.ELU(),
            nn.Linear(n_coeff + 2, n_coeff + 2),
            # nn.BatchNorm1d(n_coeff + 2, momentum=0.99, eps=0.001),
            nn.ELU(),
            nn.Linear(n_coeff + 2, n_coeff + 2),
            # nn.BatchNorm1d(n_coeff + 2, momentum=0.99, eps=0.001),
            nn.ELU(),
            nn.Linear(n_coeff + 2, 3),
            # nn.ELU()
        )

    def forward(self, embedding, l_dir):
        # l_dir =self.norm(l_dir)
        latent_code = torch.cat([embedding, l_dir], dim=-1)
        rgb = self.decoder(latent_code)
        return rgb


class NeuralRtiEncoderVae(nn.Module):
    def __init__(self, n_coeff, n_lights):
        super().__init__()
        n_units = 3 * n_lights
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(n_units, momentum=0.01, eps=0.001),
            nn.Linear(n_units, n_units * 2),
            nn.ReLU(),
            nn.Linear(n_units * 2, n_units * 2),
            nn.ReLU(),
            nn.Linear(n_units * 2, n_units * 2),
            nn.ReLU(),
            nn.Linear(n_units * 2, n_coeff * 2),
            nn.ReLU(),
        )

    def forward(self,  ray):
        embedding = self.encoder(ray)
        return embedding



class NeuralRtiDecoderVae(nn.Module):
    def __init__(self, n_coeff):
        super().__init__()
        # self.norm = nn.BatchNorm1d(2, momentum=0.01, eps=0.001)
        self.decoder = nn.Sequential(
            # nn.Linear(n_coeff + 2, n_coeff + 2),
            # nn.ReLU(),
            nn.Linear(n_coeff + 2, n_coeff + 2),
            nn.ReLU(),
            nn.Linear(n_coeff + 2, n_coeff + 2),
            nn.ReLU(),
            nn.Linear(n_coeff + 2, 3),
            # nn.Sigmoid()
        )

    def forward(self, embedding):
        # l_dir =self.norm(l_dir)
        # latent_code = torch.cat([embedding, l_dir], dim=-1)
        rgb = self.decoder(embedding)
        return rgb