import os
from train_bh import make_parser
from core.tasks import setup_task_from_args


os.environ["CUDA_VISIBLE_DEVICES"] = "7"


def main(args):
    task = setup_task_from_args(args)
    task.infer()


if __name__ == "__main__":
    parser = make_parser()

    parser.add_argument("--n-grids", nargs="+", type=int, default=[100, 100])
    parser.add_argument("--lower-bound-grids", nargs="+", type=int, default=[-2000, 1000])
    parser.add_argument("--upper-bound-grids", nargs="+", type=int, default=[3e4, 1600])
    parser.add_argument("--ckpt-file", type=str, default="ckpt_last.pt")

    args = parser.parse_args()
    main(args)
