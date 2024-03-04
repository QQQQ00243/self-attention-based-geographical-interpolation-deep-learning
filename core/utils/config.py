import os
import torch
import random
import logging
import numpy as np


def config_logging(logs_dir):
    if not os.path.exists(logs_dir):
        os.mkdir(logs_dir)
    import datetime
    log_file = datetime.datetime.now().strftime(os.path.join(logs_dir, "%Y-%m-%d_%H-%M-%S.log"))
    logging.basicConfig(
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    

def config_seed(seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
