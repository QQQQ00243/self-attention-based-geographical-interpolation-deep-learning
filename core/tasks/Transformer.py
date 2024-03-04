import os
import json
import torch
import torch.nn as nn
import logging
import numpy as np

from argparse import Namespace

from .base_task import BaseTask
from core.tasks import register_task
from core.utils import recursive_device
from core.utils import make_dirs, config_seed, config_logging


logger = logging.getLogger(__name__)


class Trainer:
    def __init__( 
        self, 
        device, 
        dataloader, 
        model, 
        optimizer, 
        lr_scheduler,
        epochs,
    ):
        self.device = device
        self.epochs = epochs
        self.train_loader = dataloader["train"]
        self.test_loader = dataloader["test"]
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    @staticmethod
    def train_one_epoch(dataloader, device, model, optimizer, lr_scheduler):
        model.train()
        loss_all = torch.empty((len(dataloader),))

        for i, inputs in enumerate(dataloader):
            recursive_device(dct=inputs, device=device)
            outputs = model(inputs)
            loss = outputs["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            loss_all[i] = loss

        return loss_all.mean().item()
    
    @staticmethod
    def eval(dataloader, device, model):
        model.eval()
        scores = torch.empty((len(dataloader),))

        with torch.no_grad():
            for i, inputs in enumerate(dataloader):
                recursive_device(dct=inputs, device=device)
                outputs = model(inputs)
                score = outputs["eval_score"]
                scores[i] = score

        return scores.mean().item()

    def train(self, weights_path):
        logger.info("Start training...")

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
            '''
            eval_score = self.eval(
                device=self.device,
                model=self.model,
                dataloader=self.test_loader, 
            )
            '''
            # print statistics
            eval_score = 0
            LR = self.lr_scheduler.get_last_lr()[0]
            logger.info("Epoch {:02d}/{:02d}, LR {:.6f} Loss {:.4f}, Eval-score {:.4f}".format(
                epoch+1, self.epochs, LR, loss, eval_score))

        logger.info("Training finished. Saving model to {}".format(weights_path))
        torch.save(self.model.state_dict(), weights_path)
        # no return


class Infer:
    def __init__(
        self,
        device,
        dataloaders,
        model,
        site_size,
        n_cls,
    ):
        self.device = device
        self.dataloaders = dataloaders
        self.model = model
        self.site_size = site_size
        self.n_cls = n_cls
    
    @staticmethod
    def get_preds(device, dataloaders, model, size, n_cls):
        model.eval()
        logits_pred = torch.empty(size + (n_cls,))
        size = torch.tensor(size).to(device)

        with torch.no_grad():
            for i, inputs in enumerate(dataloaders["train"]):
                recursive_device(inputs, device)
                trg_coord = inputs["net_input"]["trg_coord"]

                # model inference
                outputs = model(inputs)
                logits = outputs["logits"]

                # back-calculate from normalization
                trg_coord = trg_coord*size
                for coord, logit in zip(trg_coord, logits):
                    x, y = coord.round().long().tolist()
                    logits_pred[x, y, :] = logit

            scores = torch.empty((len(dataloaders["test"]),))
            for i, inputs in enumerate(dataloaders["test"]):
                recursive_device(inputs, device)
                trg_coord = inputs["net_input"]["trg_coord"]

                # model inference
                outputs = model(inputs)
                logits = outputs["logits"]
                scores[i] = outputs["eval_score"]

                # back-calculate from normalization
                trg_coord = trg_coord*size
                for coord, logit in zip(trg_coord, logits):
                    x, y = coord.round().long().tolist()
                    logits_pred[x, y, :] = logit

        labels_pred = torch.argmax(logits_pred, 2)
        return logits_pred.numpy(), labels_pred.numpy(), scores.mean().item()

    @staticmethod
    def attn_rollout(device, dataloaders, model, size, head_fusion, discard_ratio):
        from core.utils import AttentionRollout

        model.eval()
        res = {}
        size_t = torch.tensor(size).to(device)
        attn_rollout = AttentionRollout(
            model=model,
            attention_layer_name="attn",
            head_fusion=head_fusion,
            discard_ratio=discard_ratio,
        )

        with torch.no_grad():
            for i, inputs in enumerate(dataloaders["train"]):
                logger.info("{}/{}".format(i+1, len(dataloaders["train"])))
                recursive_device(inputs, device)
                trg_coord = (inputs["net_input"]["trg_coord"] * size_t).round().long()
                # (N, 2)
                site_coords = (inputs["net_input"]["site_coords"] * size_t).round().long()
                # (N, l, 2)
                attn_dict_batch = attn_rollout(inputs)
                # {0: (N, l+1, l+1), 1: (N, l+1, l+1), 2: (N, l+1, l+1)}

                # format attn_dict
                attn_dict_list = []
                for i in range(trg_coord.size(0)):
                    attn = {j: attn[i] for j, attn in attn_dict_batch.items()}
                    attn_dict_list.append(attn)

                for coord, idx, attn_dict in zip(trg_coord, site_coords, attn_dict_list):
                    '''
                    mask_dict = {}
                    for l_idx, attn in attn_dict.items():
                        # mask = torch.zeros(size)
                        # mask[idx[:, 0], idx[:, 1]] = attn
                        mask_dict[l_idx] = attn
                    '''
                    res[tuple(coord.tolist())] = {
                        "attn": attn_dict,
                        "site_coords": idx,
                    }

        return res
                
    def infer(self, logits_path, labels_path):
        logger.info("Start inferring...")

        logits_pred, labels_pred, eval_score = self.get_preds(
            device=self.device,
            dataloaders=self.dataloaders,
            model=self.model,
            size=self.site_size,
            n_cls=self.n_cls,
        )
        logger.info(f"Eval accuracy: {eval_score:.4f}")

        logger.info(f"Saving results to {logits_path}, {labels_path}")
        np.save(logits_path, logits_pred)
        np.savetxt(labels_path, labels_pred, fmt="%d")

    def get_attn_rollout(self, head_fusion, discard_ratio, res_path):
        logger.info("Start attn rollout...")
        masks_dict = self.attn_rollout(
            device=self.device,
            dataloaders=self.dataloaders,
            model=self.model,
            size=self.site_size,
            head_fusion=head_fusion,
            discard_ratio=discard_ratio,
        )
        logger.info("Saving result to {}".format(res_path))
        torch.save(masks_dict, res_path)


@register_task("Transformer")
class TransformerTask(BaseTask):
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
        self.n_bhs = 5
        self.data_root = "./data"
        self.data_file = None
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
        self.act_name = "gelu"
        self.embed_dim = 64
        self.depth = 1
        self.n_heads = 4
        self.forward_expansion = 3
        self.p = 0.1
        self.attn_p = 0.1

        # infer args
        self.head_fusion = "max"
        self.discard_ratio = 0.1
        
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
        #                 |---N6
        #                     |---logs
        #                         |---2023-04-15_11-30-12.log
        #                     |---N6.pth
        #                     |---labels.csv
        #                     |---logits.csv 
        task.save_dir = os.path.join(task.root, task.res_dir, task.task, task.task_subdir, f"N{task.n_bhs}")
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
    
    def get_loss_crit(self, name):
        if name == "CrossEntropy":
            from torch.nn import CrossEntropyLoss
            return CrossEntropyLoss()
        else:
            raise NotImplementedError(f"Unknown loss type {name}.")
        
    def get_eval_crit(self, name):
        if name == "CrossEntropy":
            from torch.nn import CrossEntropyLoss
            return CrossEntropyLoss()
        elif name == "Accuracy":
            def eval_crit(logits, labels):
                preds = torch.argmax(logits, 1)
                acc = (preds == labels).sum().item() / len(labels)
                return acc
            return eval_crit
        
    def get_model(self, device):
        from core.models import Transformer
        model = Transformer(
            device=device,
            cls_encoding_type=self.cls_encoding,
            cls_encoding_p=self.cls_encoding_p,
            pos_encoding_type=self.pos_encoding,
            pos_encoding_p=self.pos_encoding_p,
            act_name=self.act_name,
            n_cls=self.n_cls,
            site_dim=self.site_dim,
            depth=self.depth,
            embed_dim=self.embed_dim,
            n_heads=self.n_heads,
            forward_expansion=self.forward_expansion,
            p=self.p,
            attn_p=self.attn_p,
            loss_crit=self.get_loss_crit("CrossEntropy"),
            eval_crit=self.get_eval_crit("Accuracy"),
        ).to(device)
        logger.info("Model: \n{}".format(model))
        return model

    def get_dataset(self):
        from core.data import SiteDataset
        train_dataset = SiteDataset(
            train=True,
            n_bhs=self.n_bhs,
            n_cls=self.n_cls,
            shuffle_site=self.shuffle_site,
            path=os.path.join(self.data_root, self.data_file),
            normalize=True,
        )
        test_dataset = SiteDataset(
            train=False,
            n_bhs=self.n_bhs,
            n_cls=self.n_cls,
            shuffle_site=self.shuffle_site,
            path=os.path.join(self.data_root, self.data_file),
            normalize=True,
        )
        self.dataset = {
            "train": train_dataset,
            "test": test_dataset,
        }
        return self.dataset
    
    def get_training_dataloaders(self):
        from torch.utils.data import DataLoader
        dataset = self.get_dataset()
        train_loader = DataLoader(
            dataset=dataset["train"],
            batch_size=self.batch_size,
            shuffle=True,
        )
        test_loader = DataLoader(
            dataset=dataset["test"],
            batch_size=self.test_batch_size,
            shuffle=False,
        )
        self.dataloader = {
            "train": train_loader, 
            "test": test_loader,
        }
        return self.dataloader
    
    def get_optimizer_lr_scheduler(self, model: nn.Module):
        from torch.optim import Adam
        from torch.optim.lr_scheduler import LambdaLR

        optimizer = Adam(model.parameters(), lr=self.init_lr)
        total_steps = len(self.dataloader["train"]) * self.epochs
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

        '''
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # polynomial decay of order self.decay_order
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return max(0.0, 1.0 - progress) ** self.decay_order
        '''

        lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return optimizer, lr_scheduler 
    
    def train(self):
        device = self.get_device()
        dataloader = self.get_training_dataloaders()
        model = self.get_model(device)
        optimizer, lr_scheduler = self.get_optimizer_lr_scheduler(model)

        trainer = Trainer(
            device=device,
            dataloader=dataloader, 
            model=model, 
            optimizer=optimizer, 
            lr_scheduler=lr_scheduler,
            epochs=self.epochs,
        )
        weights_path = os.path.join(self.save_dir, f"N{self.n_bhs}.pth")
        trainer.train(weights_path)
        # no return
    
    def infer(self):
        device = self.get_device()
        dataloader = self.get_training_dataloaders()
        model = self.get_model(device)
        weights_path = os.path.join(self.save_dir, f"N{self.n_bhs}.pth")
        logger.info(f"Loading from {weights_path}")
        model.load_state_dict(torch.load(weights_path))

        infer = Infer(
            device=device,
            dataloaders=dataloader,
            model=model,
            n_cls=self.n_cls,
            site_size=self.dataset["train"].site.data.shape,
        )

        infer.infer(
            logits_path=os.path.join(self.save_dir, f"N{self.n_bhs}_logits.npy"), 
            labels_path=os.path.join(self.save_dir, f"N{self.n_bhs}_labels.csv"),
        )
        # no return

    def get_attn_rollout(self):
        device = self.get_device()
        dataloaders = self.get_training_dataloaders()
        model = self.get_model(device)
        weights_path = os.path.join(self.save_dir, f"N{self.n_bhs}.pth")
        logger.info(f"Loading from {weights_path}")
        model.load_state_dict(torch.load(weights_path))

        infer = Infer(
            device=device,
            dataloaders=dataloaders,
            model=model,
            n_cls=self.n_cls,
            site_size=self.dataset["train"].site.data.shape,
        )

        infer.get_attn_rollout(
            head_fusion=self.head_fusion,
            discard_ratio=self.discard_ratio,
            res_path=os.path.join(self.save_dir, "masks.pt")
        )




