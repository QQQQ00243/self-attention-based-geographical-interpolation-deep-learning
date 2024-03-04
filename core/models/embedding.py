import torch
import torch.nn as nn

from .utils import get_activation_fn


class ClsEncoderMLP(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            forward_expansion: int,
            activation: str="gelu",
            p: float=0.,
        ):
        super(ClsEncoderMLP, self).__init__()
        self.act_fn = get_activation_fn(activation=activation)
        self.drop = nn.Dropout(p)
        self.fc1 = nn.Linear(1, embed_dim)
        self.fc2 = nn.Linear(embed_dim, embed_dim*forward_expansion)
        self.fc3 = nn.Linear(embed_dim*forward_expansion, embed_dim)

    def forward(self, x: torch.Tensor):
        # x: (N, len_seq)
        x = x.unsqueeze(-1).float()
        # x: (N, len_seq, input_dim)
        x = self.drop(self.act_fn(self.fc1(x)))
        x = self.drop(self.act_fn(self.fc2(x)))
        x = self.drop(self.fc3(x))
        # (N, len_seq, embed_dim)
        return x


class ClsEmbedding(nn.Module):
    def __init__(self, embed_dim, n_cls, p=0.):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=n_cls, embedding_dim=embed_dim)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        # x: (N, len_seq)
        x = self.drop(self.embed(x))
        # (N, len_seq, embed_dim)
        return x
    

class PosEncoderMLP(nn.Module):
    def __init__(
            self,
            site_dim: int,
            embed_dim: int,
            forward_expansion: int,
            activation: str="gelu",
            p: float=0.,
        ):
        super(PosEncoderMLP, self).__init__()
        self.act_fn = get_activation_fn(activation=activation)
        self.drop = nn.Dropout(p)
        self.fc1 = nn.Linear(site_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, embed_dim*forward_expansion)
        self.fc3 = nn.Linear(embed_dim*forward_expansion, embed_dim)

    def forward(self, x: torch.Tensor):
        # x: (N, len_seq, site_dim)
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.act_fn(x)
        x = self.drop(x)

        x = self.fc3(x)
        x = self.drop(x)
        # (N, len_seq, embed_dim)
        return x


class PositionEncoderSin2D(nn.Module):
    def __init__(self, device, embed_dim: int, n: int=10000):
        super(PositionEncoderSin2D, self).__init__()
        self.device = device
        self.embed_dim = embed_dim
        self.n = n

    def forward(self, pos):
        N, len_seq, dim = pos.size()
        assert (dim == 2), "`PositionEncoderSin2D` is used for 2D encoding only."
        pos = pos.reshape(N*len_seq, 2)
        x, y = pos[:, 0].unsqueeze(1), pos[:, 1].unsqueeze(1)
        # boundary D is the smallest integer that is divisible by 2*dim,
        # where dim is the dimension of the coordinates
        D = self.embed_dim + (-self.embed_dim) % 4

        exponents = (torch.arange(0, D, 4)/D).to(self.device)
        scale_factors = self.n**exponents
        x_terms = x/scale_factors
        y_terms = y/scale_factors

        sin_x_i = (torch.arange(0, D//2, 2)).to(self.device)
        cos_x_i = sin_x_i + 1
        sin_y_i = sin_x_i + D//2
        cos_y_i = sin_y_i + 1
        pe = (torch.empty((pos.size(0), D))).to(self.device)
        pe[:, sin_x_i] = torch.sin(x_terms)
        pe[:, cos_x_i] = torch.cos(x_terms)
        pe[:, sin_y_i] = torch.sin(y_terms)
        pe[:, cos_y_i] = torch.cos(y_terms)

        pe = pe[:, :self.embed_dim].reshape(N, len_seq, self.embed_dim)
        return pe
    

if __name__ == "__main__":
    pass

    
