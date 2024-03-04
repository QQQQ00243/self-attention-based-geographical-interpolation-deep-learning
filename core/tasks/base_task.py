import os
import json
import torch
import logging

from torch.nn import Module

from argparse import Namespace

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from core.utils import make_dirs, config_seed, config_logging, recursive_device



logger = logging.getLogger(__name__)


class BaseTask(ABC):
    """Basic class for any task."""
    def __init__(self):
        self.seed = None
    
    @classmethod
    def setup(cls, args):
        """
        Set configurations from args

        Parameters
        ----------
        args, Namespace:
            argument
        """
        assert type(args) in (Namespace, dict), f"Unsupport `args` type {type(args)}. `args` must be `dict` or `argparse.Namespace`."
        args_dct = vars(args) if isinstance(args, Namespace) else args

        # initialize task
        task = cls()
        # set attributes from args
        for key, val in args_dct.items():
            setattr(task, key, val)
        
        # make directories
        # files are organized as follows:
        # |---root
        #     |---core
        #     |---results
        #         |---task_name
        #             |---task1
        #                  |---logs
        #                      |---2023-04-15_11-30-12.log
        #                  |---N6.pth
        #                  |---labels.csv
        #                  |---logits.csv 
        task.save_dir = os.path.join(task.root, task.res_dir, task.task, task.task_subdir)
        task.logs_dir = os.path.join(task.save_dir, "logs")
        make_dirs(
            os.path.join(task.root, task.res_dir),
            os.path.join(task.root, task.res_dir, task.task),
            os.path.join(task.root, task.res_dir, task.task, task.task_subdir),
            task.save_dir,
            task.logs_dir,
        )
        
        # configurate logger
        config_logging(task.logs_dir)
        # configurate random seed
        config_seed(task.seed)

        # print arguments
        logger.info(f"Configurations: \n{task}")

        # save configurations to json file
        config_path = os.path.join(task.save_dir, "configs.json")
        logger.info("Saving configurations to {}".format(config_path))
        with open(config_path, "w") as f:
            json.dump(args_dct, indent=4, fp=f)
            f.close()
        return task
    
    def get_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using {device}.")
        return device

    @abstractmethod
    def get_loss_crit(self, name):
        pass

    @abstractmethod
    def get_eval_crit(self, name):
        pass

    @abstractmethod
    def get_model(self) -> Module:
        pass
    
    '''
    @abstractmethod
    def get_dataset(self):
        pass
        
    @abstractmethod
    def get_training_dataloaders(self):
        pass
        
    @abstractmethod
    def get_optimizer(self) -> torch.optim.Optimizer:
        pass
        
    @abstractmethod
    def get_lr_scheduler(self) -> LRScheduler:
        pass
    '''
    
    def __repr__(self):
        import pprint
        from tabulate import tabulate

        table_header = ["keys", "values"]
        exp_table = [
            (str(k), pprint.pformat(v))
            for k, v in vars(self).items()
            if not k.startswith("_")
        ]
        return tabulate(exp_table, headers=table_header, tablefmt="fancy_grid")


if __name__ == "__main__":
    pass