from train import make_parser
from core.tasks import setup_task_from_args


def main(args):
    task = setup_task_from_args(args)
    # task.infer()
    task.get_attn_rollout()


if __name__ == "__main__":
    parser = make_parser()
    '''
    parser.add_argument("--n-grids", nargs="+", type=int, default=[300, 300])
    parser.add_argument("--lower-bound-test", nargs="+", type=int, default=[0, 900])
    parser.add_argument("--upper-bound-test", nargs="+", type=int, default=[25000, 1600])
    parser.add_argument("--ckpt-file", type=str, default="ckpt_best.pt")
    '''

    parser.add_argument("--head-fusion", type=str, default="mean", choices=["min", "max", "mean"])
    parser.add_argument("--discard-ratio", type=float, default=0)

    args = parser.parse_args()
    main(args)
