import os
from core.tasks import setup_task_from_file


if __name__ == "__main__":
    root = "/mnt/workspace/qiyahang/GIT/GIT-V10/"
    file = "results/Transformer/exp2/N6/configs.json"
    task = setup_task_from_file(os.path.join(root, file))
    # task.infer()
    task.get_attn_rollout()
    

