import os
import json
import logging
import numpy as np
import torch
import torch.nn as nn

from argparse import Namespace

from .base_task import BaseTask
from core.data import Boreholes, TransformerDataset
from core.tasks import register_task
from core.utils import make_dirs, recursive_device, config_seed, config_logging

from torch.utils.data import DataLoader


logger = logging.getLogger(__name__)


class CheckpointChecker:
    def __init__(self, mode='max'):
        assert (mode in ("min", "max")), f"Unsupported value for `mode`, must be 'min' or 'max'"
        self.mode = mode
        self.best_score = None

    def __call__(self, eval_score):
        if self.best_score is None:
            self.best_score = eval_score
        is_best = False
        if (self.mode == 'max') and (eval_score > self.best_score):
                self.best_score = eval_score
                is_best = True
        elif (self.mode == 'min') and (eval_score < self.best_score):
                self.best_score = eval_score
                is_best = True
        return is_best


class Trainer:
    def __init__( 
        self, 
        device, 
        dataloaders, 
        model, 
        optimizer, 
        lr_scheduler,
        epochs,
        train_valid_split
    ):
        self.device = device
        self.epochs = epochs
        self.train_loader = dataloaders["train"]
        self.valid_loader = dataloaders["valid"]
        self.train_valid_split = train_valid_split
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    @staticmethod
    def train_one_epoch(dataloader, device, model, optimizer, lr_scheduler):
        model.train()
        loss_sum = 0
        n_samples = 0

        for i, inputs in enumerate(dataloader):
            recursive_device(dct=inputs, device=device)
            outputs = model(inputs)
            loss = outputs["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            loss_sum += loss.item()
            n_samples += inputs["target"].size(0)

        return loss_sum / n_samples
    
    @staticmethod
    def eval(dataloader, device, model):
        model.eval()
        score_sum = 0
        n_samples = 0

        with torch.no_grad():
            for i, inputs in enumerate(dataloader):
                recursive_device(dct=inputs, device=device)
                outputs = model(inputs)
                score = outputs["eval_score"]
                # score_sum += score.item()
                score_sum += score
                n_samples += inputs["target"].size(0)

        return score_sum / n_samples

    def save(self, path):
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "train_valid_split": self.train_valid_split,
            },
            path,
        )
        # no return

    def train(self, save_dir):
        logger.info("Start training...")
        ckpt_path_best = os.path.join(save_dir, "ckpt_best.pt")
        ckpt_path_last = os.path.join(save_dir, "ckpt_last.pt")
        ckpt_checker = CheckpointChecker(mode="max")

        for epoch in range(self.epochs):
            # train for one epoch
            loss = self.train_one_epoch(
                device=self.device,
                model=self.model,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                dataloader=self.train_loader,
            )
            # evaluate on test dataset
            eval_score = self.eval(
                device=self.device,
                model=self.model,
                dataloader=self.valid_loader, 
            )
            # print statistics
            LR = self.lr_scheduler.get_last_lr()[0]
            logger.info("Epoch {:02d}/{:02d}, LR {:.6f} Loss {:.4f}, Eval-score {:.4f}".format(
                epoch+1, self.epochs, LR, loss, eval_score))
            is_best = ckpt_checker(eval_score=eval_score)
            if is_best:
                logger.info("Best eval score encountered. Saving weights to {}".format(ckpt_path_best))
                self.save(path=ckpt_path_best)
            
        logger.info(f"Training finished. Saving weights of last epoch to {ckpt_path_last}")
        self.save(path=ckpt_path_last)
        # no return


class Infer:
    def __init__(
        self,
        device,
        save_dir,
        dataloaders,
        model,
        bhs: Boreholes,
    ):
        self.device = device
        self.save_dir = save_dir
        self.bhs = bhs
        self.test_loader = dataloaders["test"]
        self.grid_loader = dataloaders["grid"]
        self.model = model

    @staticmethod
    def eval(dataloader, device, model):
        model.eval()
        score_sum = 0
        n_samples = 0

        with torch.no_grad():
            for i, inputs in enumerate(dataloader):
                recursive_device(dct=inputs, device=device)
                outputs = model(inputs)
                score = outputs["eval_score"]
                score_sum += score
                n_samples += inputs["target"].size(0)

        return score_sum / n_samples

    def infer_grid(self):
        self.model.eval()

        # initialize predicted logits `logits_pred`
        logits_pred = np.empty(self.bhs.n_grids.tolist())

        # load bound of normalization
        site_size_norm = self.bhs.size_norm
        lower_bound_norm = self.bhs.lower_bound_norm

        # load lower bound of grids and grid size to compute indicies
        lower_bound_test = self.bhs.lower_bound_test
        grid_size = self.bhs.grid_size

        # ------- compute mask for grids ------- #
        coords_grids = self.bhs.data_grids["coords"].numpy()
        coords_grids_un = coords_grids*site_size_norm + lower_bound_norm

        # if `sub_depth` is True, compute depth from horizontal coordinates
        # and add depth to vertical coordinates
        if self.bhs.sub_depth:
            depth = np.interp(coords_grids_un[:, 0], self.bhs.vertices[:, 0], self.bhs.vertices[:, 1])
            # un-normalize
            coords_grids_un[:, 1] += depth
        # compute indices of selected grids from coordinates
        mask_indices = ((coords_grids_un - lower_bound_test - grid_size/2) / grid_size).round().astype(int)

        # initialize mask
        mask = np.zeros(logits_pred.shape, dtype=bool)
        # set items of mask at selected grids to True
        mask[mask_indices[:, 0], mask_indices[:, 1]] = True
        # -------------------------------------- #

        # ------- do inference and save result to `logits_pred` ------- #
        with torch.no_grad():
            for i, inputs in enumerate(self.grid_loader):
                recursive_device(inputs, self.device)

                # model inference
                outputs = self.model(inputs)

                # operate using numpy
                trg_coord = inputs["net_input"]["trg_coord"].cpu().numpy()
                logits_batch = outputs["logits"].cpu().numpy()

                # unnormalize
                trg_coord_un = trg_coord*site_size_norm + lower_bound_norm

                # if `sub_depth` is True, compute depth from horizontal coordinates
                # and add depth to vertical coordinates
                if self.bhs.sub_depth:
                    depth = np.interp(trg_coord_un[:, 0], self.bhs.vertices[:, 0], self.bhs.vertices[:, 1])
                    trg_coord_un[:, 1] += depth
                # compute indices of selected grids from coordinates
                indices = ((trg_coord_un - lower_bound_test - grid_size/2) / grid_size).round().astype(int)
                # save inference result to `logits_pred`
                for idx, logit in zip(indices, logits_batch):
                    i, j = idx
                    logits_pred[i, j] = logit
        # ------------------------------------------------------------ #
        
        labels_pred = np.copy(logits_pred)
        positive_mask = logits_pred > 0
        labels_pred[positive_mask & mask] = 1
        labels_pred[~positive_mask & mask] = 0
        labels_pred[~mask] = -1

        return logits_pred, labels_pred, mask

    def infer(self):
        logger.info("Start inferring...")

        logits_path=os.path.join(self.save_dir, f"logits.csv")
        labels_path=os.path.join(self.save_dir, f"labels.csv")
        mask_path=os.path.join(self.save_dir, f"mask.csv")

        # compute score on validation set
        if self.test_loader is not None:
            test_score = self.eval(
                device=self.device,
                dataloader=self.test_loader,
                model=self.model,
            )
            logger.info(f"Test score: {test_score:.3f}")

        # infer on test set
        logits_pred, labels_pred, mask = self.infer_grid()

        logger.info(f"Saving results to {self.save_dir}")
        np.savetxt(logits_path, logits_pred)
        np.savetxt(labels_path, labels_pred, fmt="%d")
        np.savetxt(mask_path, mask, fmt="%d")


@register_task("TransformerBH")
class TransformerBHTask(BaseTask):
    def __init__(self):
        super().__init__()
        # ------ task configs ------ #
        self.root = "/nvme/qiyahang/GIT/GIT-V3"
        self.task = None
        self.res_dir = "results"
        self.task_subdir = None
        self.save_dir = None
        self.logs_dir = None

        # ------ data ------ #
        self.data_root = "./data"
        self.data_file = None
        self.valid_size = 0.1
        self.threshold = 0.06

        self.test_site = None

        self.shuffle_site = True
        self.batch_size = 64
        self.test_batch_size = 256

        # ------ training ------ #
        self.epochs = 100
        self.init_lr = 1e-4
        self.warmup_ratio = 0.05
        self.decay_order = 2
        self.betas = [0.9, 0.98]
        self.eps = 1e-09

        # ------ model ------ #
        self.cls_encoding = "Embedding"
        self.cls_encoding_p = 0.1
        self.pos_encoding = "MLP"
        self.pos_encoding_p = 0.1
        self.n_cls = 4
        self.site_dim = 2
        self.activation = "gelu"
        self.embed_dim = 64
        self.depth = 1
        self.n_heads = 4
        self.forward_expansion = 3
        self.p = 0.1
        self.attn_p = 0.1
        self.ckpt_file = None

        # ------ infer ------ #
        self.n_grids = None
        self.lower_bound_grids = None
        self.upper_bound_grids = None

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
        #     |---res_dir
        #         |---task
        #             |---task_subdir
        #                  |---test_site
        #                      |---logs
        #                          |---2023-04-15_11-30-12.log
        #                      |---ckpt_last.pth
        #                      |---ckpt_best.pth
        #                      |---labels.csv
        #                      |---logits.csv
        test_site = task.test_site if task.test_site is not None else "all"
        task.save_dir = os.path.join(task.root, task.res_dir, task.task, task.task_subdir, test_site)
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

    def get_loss_crit(self):
        from torch.nn import BCEWithLogitsLoss
        def loss_crit(logits, targets):
            loss = BCEWithLogitsLoss(reduction="sum")
            return loss(logits.view_as(targets), targets.float())
        return loss_crit
        
    def get_eval_crit(self, name):
        if name == "BCE":
            from torch.nn import BCEWithLogitsLoss
            return BCEWithLogitsLoss()
        elif name == "Accuracy":
            def eval_crit(logits, labels):
                preds = (logits.view_as(labels) > 0).long()
                acc = (preds == labels).sum()
                return acc
            return eval_crit
        elif name == "F1-score":
            class F1_score:
                def __init__(self, threshold=0.5, eps=1e-7):
                    self.threshold = threshold
                    self.eps = eps

                def __call__(self, logits, targets):
                    preds = (torch.sigmoid(logits) >= self.threshold).float()
                    TP = (preds * targets).sum()
                    pred_positives = preds.sum()
                    actual_positives = targets.sum()

                    precision = TP / (pred_positives + self.eps)
                    recall = TP / (actual_positives + self.eps)
                    f1_score = 2 * (precision * recall) / (precision + recall + self.eps)

                    return f1_score
            return F1_score()
        raise NotImplementedError("Eval criterion not implemented")
        
    def get_model(self, device):
        from core.models import Transformer
        model = Transformer(
            device=device,
            cls_encoding_type=self.cls_encoding,
            cls_encoding_p=self.cls_encoding_p,
            pos_encoding_type=self.pos_encoding,
            pos_encoding_p=self.pos_encoding_p,
            n_cls=self.n_cls,
            site_dim=self.site_dim,
            depth=self.depth,
            loss_crit=self.get_loss_crit(),
            eval_crit=self.get_eval_crit("Accuracy"),
            act_name="gelu",
            embed_dim=self.embed_dim,
            n_heads=self.n_heads,
            forward_expansion=self.forward_expansion,
            p=self.p,
            attn_p=self.attn_p,
        ).to(device)
        logger.info("Model: \n{}".format(model))
        return model

    def get_training_datasets(self, sites):
        bhs = Boreholes(os.path.join(self.data_root, self.data_file))
        train_valid_split = bhs.split_train_valid(
            valid_size=self.valid_size,
            sites=sites,
            normalize=True, 
            return_tensor=True,
        )
        train_set = TransformerDataset(
            data_train=train_valid_split["train"],
            train=True, 
            shuffle_site=True,
        )
        valid_set = TransformerDataset(
            data_train=train_valid_split["train"],
            data_test=train_valid_split["valid"],
            train=False, 
            shuffle_site=True,
        )
        self.datasets = {"train": train_set, "valid": valid_set, "Boreholes": bhs}
        return self.datasets
    
    def get_infer_datasets(self, train_valid_split, sites=None):
        # get validation set
        valid_set = TransformerDataset(
            data_train=train_valid_split["train"],
            data_test=train_valid_split["valid"],
            train=False, 
            shuffle_site=True,
        )

        bhs = Boreholes(path=os.path.join(self.data_root, self.data_file))
        bhs.train_valid_split = train_valid_split
        
        # get test_set if `sites` is not None
        test_set = None
        if sites is not None:
            data_test = bhs.get_test(sites=sites, normalize=True, return_tensor=True)
            test_set = TransformerDataset(
                data_train=train_valid_split["train"],
                data_test=data_test,
                train=False,
                shuffle_site=True,
            )

        # get grids set
        data_grids = bhs.generate_grids(
            n_grids=self.n_grids,
            lower_bound=self.lower_bound_grids,
            upper_bound=self.upper_bound_grids,
            normalize=True,
            return_tensor=True,
        )
        grid_set = TransformerDataset(
            data_train=train_valid_split["train"],
            data_test=data_grids,
            train=False,
            shuffle_site=True,
        )
        self.infer_datasets = {"valid": valid_set, "test": test_set, "grid": grid_set, "Boreholes": bhs}
        return self.infer_datasets
    
    def get_training_loaders(self, sites):
        from core.utils import ImbalancedDatasetSampler
        datasets = self.get_training_datasets(sites=sites)
        '''
        train_loader = DataLoader(
            dataset=datasets["train"],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        '''
        train_loader = DataLoader(
            dataset=datasets["train"],
            sampler=ImbalancedDatasetSampler(datasets["train"]),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        valid_loader = DataLoader(
            dataset=datasets["valid"],
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        self.dataloaders = {"train": train_loader, "valid": valid_loader}
        return self.dataloaders
    
    def get_infer_loaders(self, train_valid_split, sites=None):
        infer_datasets = self.get_infer_datasets(train_valid_split=train_valid_split, sites=sites)

        test_loader = None
        if sites is not None:
            test_loader = DataLoader(
                dataset=infer_datasets["test"],
                batch_size=self.test_batch_size,
                shuffle=False,
            )

        grid_loader = DataLoader(
            dataset=infer_datasets["grid"],
            batch_size=self.test_batch_size,
            shuffle=False,
        )
        self.infer_dataloaders = {
            "test": test_loader,
            "grid": grid_loader,
        }
        return self.infer_dataloaders
    
    def get_optimizer_lr_scheduler(self, model: nn.Module):
        from torch.optim import Adam
        from torch.optim.lr_scheduler import LambdaLR

        optimizer = Adam(model.parameters(), lr=self.init_lr)
        total_steps = len(self.dataloaders["train"]) * self.epochs
        warmup_steps = int(self.warmup_ratio * total_steps)

        class LRLambda:
            def __init__(self, total_steps, warmup_steps, decay_order):
                self.warmup_steps = warmup_steps
                self.decay_steps = total_steps - warmup_steps
                self.decay_order = decay_order

            def __call__(self, current_step):
                if current_step < warmup_steps:
                    # Linear warmup
                    return float(current_step) / float(max(1, warmup_steps))
                else:
                    # polynomial decay of order self.decay_order
                    progress = current_step - warmup_steps
                    decay_factor =  ((1.0 - progress/self.decay_steps) / (1.0 - (progress-1)/self.decay_steps)) ** self.decay_order
                    return decay_factor
                
        lr_lambda = LRLambda(total_steps=total_steps, warmup_steps=warmup_steps, decay_order=self.decay_order)

        lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return optimizer, lr_scheduler 
    
    def train(self):
        device = self.get_device()

        bhs = Boreholes(os.path.join(self.data_root, self.data_file))
        site_list = list(bhs.bhs_dict.keys())
        train_sites = site_list.copy()

        # remove site for testing if `self.test_site` is not None
        if self.test_site is not None:
            for test_site_i in self.test_site.split(","):
                train_sites.remove(test_site_i)

        datasets = self.get_training_datasets(sites=train_sites)
        bhs = datasets["Boreholes"]
        dataloaders = self.get_training_loaders(sites=train_sites)
        model = self.get_model(device)
        optimizer, lr_scheduler = self.get_optimizer_lr_scheduler(model)

        trainer = Trainer(
            device=device,
            dataloaders=dataloaders, 
            model=model, 
            optimizer=optimizer, 
            lr_scheduler=lr_scheduler,
            epochs=self.epochs,
            train_valid_split=bhs.train_valid_split,
        )
        # ckpt_path_best = os.path.join(self.save_dir, f"ckpt_best.pth")
        trainer.train(self.save_dir)
        # no return
    
    def infer(self):
        device = self.get_device()
        model = self.get_model(device)

        # load ckpt
        ckpt_path = os.path.join(self.save_dir, self.ckpt_file)
        logger.info(f"Loading from {ckpt_path}")
        ckpt_dict = torch.load(ckpt_path)
        model.load_state_dict(ckpt_dict["state_dict"])
        train_valid_split = ckpt_dict["train_valid_split"]

        # test-sites
        if self.test_site is not None:
            test_sites = self.test_site.split(",")
        else:
            test_sites = None
        # test_sites = [self.test_site] if self.test_site is not None else None
        infer_datasets = self.get_infer_datasets(
            train_valid_split=train_valid_split,
            sites=test_sites,
        )
        dataloaders = self.get_infer_loaders(
            train_valid_split=train_valid_split,
            sites=test_sites,
        )

        infer = Infer(
            device=device,
            dataloaders=dataloaders,
            model=model,
            bhs=infer_datasets["Boreholes"],
            save_dir=self.save_dir,
        )

        infer.infer()
        # no return


