import os
import torch
import torch.nn as nn

from .utils import get_activation_fn

from .block import Block
from .embedding import ClsEncoderMLP, ClsEmbedding, PosEncoderMLP, PositionEncoderSin2D

    
class Transformer(nn.Module):
    def __init__(
            self,
            device,
            cls_encoding_type: str,
            cls_encoding_p: float,
            pos_encoding_type: str,
            pos_encoding_p: float,
            n_cls: int,
            site_dim: int,
            depth: int,
            loss_crit,
            eval_crit,
            act_name: str="gelu",
            embed_dim: int=768,
            n_heads: int=4,
            forward_expansion: int=4,
            p: float=0.1,
            attn_p: float=0.1,
        ):
        super().__init__()

        # ------- input encoding ------- #
        if cls_encoding_type == "MLP":
            self.cls_encoder = ClsEncoderMLP(
                p=cls_encoding_p,
                embed_dim=embed_dim,
                activation=act_name,
                forward_expansion=forward_expansion,
            )
        elif cls_encoding_type == "Embedding":
            self.cls_encoder = ClsEmbedding(
                n_cls=n_cls,
                p=cls_encoding_p,
                embed_dim=embed_dim,
            )
        else:
            raise ValueError("Unknown input embedding type {}".format(cls_encoding_type))
        
        # ------- positional encoding ------- #
        if pos_encoding_type == "sin":
            self.pos_encoder = PositionEncoderSin2D(embed_dim=embed_dim, device=device)
        elif pos_encoding_type == "MLP":
            self.pos_encoder = PosEncoderMLP(
                p=pos_encoding_p,
                site_dim=site_dim,
                embed_dim=embed_dim,
                activation=act_name,
                forward_expansion=forward_expansion
            )
        else:
            raise ValueError("Unknown positional encoding type {}".format(pos_encoding_type))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_drop = nn.Dropout(p=p)

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim=embed_dim,
                    n_heads=n_heads,
                    forward_expansion=forward_expansion,
                    attn_p=attn_p,
                    p=p,
                    act_name=act_name,
                )
                for _ in range(depth)
            ]
        )
        
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_cls if n_cls > 2 else 1)

        self.loss_crit = loss_crit
        self.eval_crit = eval_crit

    def forward(self, inputs):
        net_input = inputs["net_input"]
        site_coords = net_input["site_coords"]
        # site_coords: (N, len_seq, site_dim)
        site_cls = net_input["site_cls"]
        # site_cls: (N, len_seq)
        trg_coord = net_input["trg_coord"]
        # trg_coord: (N, site_dim)
        
        cls_encoding = self.cls_encoder(site_cls)
        # (N, len_seq, embed_dim)
        cls_token = self.cls_token.expand(
            cls_encoding.size(0), -1, -1
        )  # (N, 1, embed_dim)
        cls_token = self.cls_drop(cls_token)
        x = torch.cat((cls_token, cls_encoding), dim=1)
        # (N, 1 + len_seq, embed_dim)

        pos = torch.concat([trg_coord.unsqueeze(1), site_coords], dim=1)
        # (N, 1 + len_seq, embed_dim)
        pos_encoding = self.pos_encoder(pos)
        # (N, 1 + len_seq, embed_dim)
        x = x + pos_encoding
        # (N, 1 + len_seq, embed_dim)


        for i, block in enumerate(self.blocks):
            x = block(x)
                
        # x, size: (N, 1 + len_seq, embed_dim)

        x = self.norm(x)
        # (N, 1 + len_seq, embed_dim)
        cls_token_final = x[:, 0]  # just the CLS token
        # (N, embed_dim)
        logits = self.head(cls_token_final)
        # (N, n_cls)

        trg_cls = inputs["target"]
        # trg_cls: (N)
        if self.training:
            loss = self.loss_crit(logits, trg_cls)
            return {
                "logits": logits,
                "loss": loss,
            }
        else:
            if trg_cls is not None:
                eval_score = self.eval_crit(logits, trg_cls)
            else:
                eval_score = None
            return {
                "logits": logits,
                "eval_score": eval_score,
            }


if __name__ == "__main__":
    pass
