import os
import logging
import argparse

from core.tasks import setup_task_from_args


logger = logging.getLogger(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = "6"


def make_parser():
    parser = argparse.ArgumentParser()
    # --------------------- exp configs --------------------- # 
    parser.add_argument("--root", type=str, default="/mnt/workspace/qiyahang/GIT/GIT-V14")
    parser.add_argument("--task", type=str, default="Transformer")
    parser.add_argument("--task_subdir", type=str, default="exp5")
    
    # --------------------- data --------------------- #
    parser.add_argument("--n_bhs", type=int, default=10)
    parser.add_argument("--data_root", type=str, default="/mnt/workspace/qiyahang/GIT/dataset")
    parser.add_argument("--data_file", type=str, default="TI_reduced_final.csv")
    parser.add_argument("--valid-size", type=float, default=0.1)
    parser.add_argument("--shuffle_site", action="store_true", default=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test-batch_size", type=int, default=128)
    
    # --------------------- training --------------------- #
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--init_lr", type=float, default=3e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--decay_order", type=float, default=200.0)
    parser.add_argument("--betas", nargs="+", type=float, default=[0.9, 0.98])
    parser.add_argument("--eps", type=float, default=1e-9)

    # --------------------- model --------------------- #
    parser.add_argument("--cls_encoding", choices=["MLP", "Embedding"], type=str, default="Embedding")
    parser.add_argument("--cls_encoding_p", type=float, default=0.05)
    parser.add_argument("--pos_encoding", choices=["MLP", "sin"], type=str, default="MLP")
    parser.add_argument("--pos_encoding_p", type=float, default=0.05)
    parser.add_argument("--n_cls", type=int, default=4)
    parser.add_argument("--site_dim", type=int, default=2)
    parser.add_argument("--act-name", choices=["relu", "gelu", "tanh", "linear"], type=str, default="gelu")
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=1)
    parser.add_argument("--forward_expansion", type=int, default=3)
    parser.add_argument("--p", type=float, default=0.1)
    parser.add_argument("--attn_p", type=float, default=0.1)

    return parser


def main(args):
    task = setup_task_from_args(args)
    task.train()


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
