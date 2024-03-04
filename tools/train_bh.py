import os
import logging
import argparse

from core.tasks import setup_task_from_args


logger = logging.getLogger(__name__)


def make_parser():
    parser = argparse.ArgumentParser()
    # --------------------- device --------------------- #
    parser.add_argument("--gpu-id", type=int, default=0)

    # --------------------- exp configs --------------------- # 
    parser.add_argument("--root", type=str, default="/mnt/workspace/qiyahang_1/GIT/GIT-V17")
    parser.add_argument("--task", type=str, default="TransformerBH")
    parser.add_argument("--task-subdir", type=str, default="exp1")
    
    # --------------------- data --------------------- #
    parser.add_argument("--data-root", type=str, default="/mnt/workspace/qiyahang_1/GIT/dataset")
    parser.add_argument("--data-file", type=str, default="bhs_all_0621.csv")
    parser.add_argument("--valid-size", type=float, default=0.05)
    parser.add_argument("--shuffle-site", action="store_true", default=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--test-batch-size", type=int, default=32)
    parser.add_argument("--test-site", default="W11,SH7")
    
    # --------------------- training --------------------- #
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--init-lr", type=float, default=3e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.08)
    parser.add_argument("--decay-order", type=float, default=400)
    parser.add_argument("--betas", nargs="+", type=float, default=[0.9, 0.98])
    parser.add_argument("--eps", type=float, default=1e-9)

    # --------------------- model --------------------- #
    parser.add_argument("--cls-encoding", choices=["MLP", "Embedding"], type=str, default="Embedding")
    parser.add_argument("--cls-encoding_p", type=float, default=0.05)
    parser.add_argument("--pos-encoding", choices=["MLP", "sin"], type=str, default="MLP")
    parser.add_argument("--pos-encoding-p", type=float, default=0.05)
    parser.add_argument("--n-cls", type=int, default=2)
    parser.add_argument("--site-dim", type=int, default=2)
    parser.add_argument("--act-name", choices=["relu", "gelu", "tanh", "linear"], type=str, default="gelu")
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--forward-expansion", type=int, default=3)
    parser.add_argument("--p", type=float, default=0.1)
    parser.add_argument("--attn-p", type=float, default=0.1)

    return parser


def main(args):
    task = setup_task_from_args(args)
    task.train()


if __name__ == "__main__":
    args = make_parser().parse_args()

    # set cuda device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    main(args)
