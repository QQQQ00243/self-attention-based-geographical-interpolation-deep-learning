import torch
import torch.nn as nn

import numpy as np


class PositionEncoderSin2D1(nn.Module):
    def __init__(self, device, embed_dim: int, n: int=10000):
        super(PositionEncoderSin2D1, self).__init__()
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
    

class PositionEncoderSin2D2(nn.Module):
    def __init__(self, embed_dim: int, n: int=10000):
        super(PositionEncoderSin2D2, self).__init__()
        self.embed_dim = embed_dim
        self.n = n

    def forward(self, pos):
        pos = pos.cpu()
        N, len_seq, dim = pos.size()
        assert (dim == 2), "`PositionEncoderSin2D` is used for 2D encoding only."
        pos = pos.reshape(N*len_seq, 2)
        x, y = pos[:, 0].unsqueeze(1), pos[:, 1].unsqueeze(1)
        # boundary D is the smallest integer that is divisible by 2*dim,
        # where dim is the dimension of the coordinates
        D = self.embed_dim + (-self.embed_dim) % 4

        exponents = (torch.arange(0, D, 4)/D)
        scale_factors = self.n**exponents
        x_terms = x/scale_factors
        y_terms = y/scale_factors

        sin_x_i = (torch.arange(0, D//2, 2))
        cos_x_i = sin_x_i + 1
        sin_y_i = sin_x_i + D//2
        cos_y_i = sin_y_i + 1
        pe = (torch.empty((pos.size(0), D)))
        pe[:, sin_x_i] = torch.sin(x_terms)
        pe[:, cos_x_i] = torch.cos(x_terms)
        pe[:, sin_y_i] = torch.sin(y_terms)
        pe[:, cos_y_i] = torch.cos(y_terms)

        pe = pe[:, :self.embed_dim].reshape(N, len_seq, self.embed_dim)
        return pe
    

def test_speed(device, n=30):
    coords = torch.rand((64, 256, 2)).to(device)
    
    pe1 = PositionEncoderSin2D1(embed_dim=256, device=device)
    pe2 = PositionEncoderSin2D2(embed_dim=256)

    def test_single(pe, coords, n):
        import time
        t0 = time.time()
        for _ in range(n):
            positional_encoding = pe(coords)
        time_elapsed = time.time() - t0
        return time_elapsed

    t1 = test_single(pe=pe1, coords=coords, n=n)
    t2 = test_single(pe=pe2, coords=coords, n=n)

    print("t1: {:.3f}, t2: {:.3f}".format(t1, t2))
    return t1, t2 
    

if __name__ == "__main__":
    device = torch.device("cuda")

    '''
    t = torch.tensor([[1, 1], [1, 2]]).unsqueeze(0).to(device)
    pe = PositionEncoderSin2D1(embed_dim=16, device=device)
    positional_encoding = pe(t)
    print(positional_encoding)
    '''
    test_speed(device=device)
