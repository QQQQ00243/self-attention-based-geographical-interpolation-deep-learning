import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import get_activation_fn


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, attn_p):
        super(MultiheadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.n_heads = n_heads

        self.head_dim = embed_dim // n_heads
        assert (
            self.head_dim*n_heads == self.embed_dim
        ), "embed_dim={} must be divisible by n_heads={}".format(embed_dim, n_heads)
        self.scaling = self.head_dim**(-0.5)

        self.in_proj = nn.Linear(embed_dim, embed_dim*3)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(attn_p)

    def forward(self, x):
        bsz, seq_len, embed_dim = x.size()

        # Project input `x`` to query, key, and value
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        q = (
            q.view(bsz, seq_len, self.n_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
            .view(bsz*self.n_heads, -1, self.head_dim)
            *self.scaling
        )
        if k is not None:
            k = (
                k.view(bsz, -1, self.n_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
                .view(bsz*self.n_heads, -1, self.head_dim)
            )
        if v is not None:
            v = (
                v.view(bsz, -1, self.n_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
                .view(bsz*self.n_heads, -1, self.head_dim)
            )

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.dropout(F.softmax(attn_weights, dim=-1))

        o = torch.bmm(attn_weights, v)
        o = (
            o.view(bsz, self.n_heads, seq_len, self.head_dim)
            .transpose(1, 2)
            .contiguous()
            .view(bsz, seq_len, embed_dim)
        )
        o = self.output_proj(o)

        return o, attn_weights
    

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, act_name, out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = get_activation_fn(act_name)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        x = self.fc1(x) # (n_samples, n_patches + 1, hidden_features)
        x = self.act(x)  # (n_samples, n_patches + 1, hidden_features)
        x = self.drop(x)  # (n_samples, n_patches + 1, hidden_features)
        x = self.fc2(x)  # (n_samples, n_patches + 1, out_features)
        x = self.drop(x)  # (n_samples, n_patches + 1, out_features)

        return x
    

class Block(nn.Module):
    def __init__(
            self, 
            embed_dim, 
            n_heads, 
            forward_expansion,
            act_name,
            p=0., 
            attn_p=0.,
        ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = MultiheadAttention(
                embed_dim,
                n_heads=n_heads,
                attn_p=attn_p,
        )
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        hidden_features = int(embed_dim * forward_expansion)
        self.mlp = MLP(
                in_features=embed_dim,
                hidden_features=hidden_features,
                out_features=embed_dim,
                act_name=act_name,
        )

    def forward(self, x):
        attn_outputs, attn_weights = self.attn(x)
        x = x + self.norm1(attn_outputs)
        x = x + self.mlp(self.norm2(x))

        return x


if __name__ == "_main__":
    attn = MultiheadAttention(
        embed_dim=16,
        n_heads=8,
        attn_p=0.1,
    )
    