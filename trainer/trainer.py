import datetime
import os
import time

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets import DataManager
from evaluator import build_evaluator
from utils import AverageMeter, MetricMeter


class Trainer:
    """Generic Trainer Class for Implementing Generic Function"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.max_epoch = cfg.OPTIM.MAX_EPOCH
        self.output_dir = cfg.OUTPUT_DIR
        self.device = torch.cuda.current_device()

        self._writer = None

        # Build Data Manager
        self.data_manager = DataManager(self.cfg)
        self.data_loader_train = self.data_manager.data_loader_train
        self.data_loader_test = self.data_manager.data_loader_test
        self.num_classes = self.data_manager.num_classes
        self.len_query = self.data_manager.len_query

        self._models = OrderedDict()
        self._optimizers = OrderedDict()
        self._lr_schedulers = OrderedDict()

        # Build Model
        self.build_model()

        # Build Evaluator
        self.evaluator = build_evaluator(
            cfg, self.len_query
        )

    def build_model(self):
        raise NotImplementedError

    def set_model_mode(self, mode="train", model_names=None):
        model_names = self.get_model_names(model_names)

        for model_name in model_names:
            if mode == "train":
                self._models[model_name].train()
            elif mode in ["test", "eval"]:
                self._models[model_name].eval()
            else:
                raise KeyError

    def update_lr(self, model_names=None):
        model_names = self.get_model_names(model_names)

        for model_name in model_names:
            if self._lr_schedulers[model_name] is not None:
                self._lr_schedulers[model_name].step()

    def detect_abnormal_loss(self, loss):
        if not torch.isfinite(loss).all():
            raise FloatingPointError("Loss is Infinite or NaN.")

    def model_zero_grad(self, model_names=None):
        model_names = self.get_model_names(model_names)
        for model_name in model_names:
            if self._optimizers[model_name] is not None:
                self._optimizers[model_name].zero_grad()

    def model_backward(self, loss):
        self.detect_abnormal_loss(loss)
        loss.backward()

    def model_update(self, model_names=None):
        model_names = self.get_model_names(model_names)
        for model_name in model_names:
            if self._optimizers[model_name] is not None:
                self._optimizers[model_name].step()

    def model_backward_and_update(self, loss, model_names=None):
        self.model_zero_grad(model_names)
        self.model_backward(loss)
        self.model_update(model_names)

    def init_writer(self, log_dir):
        if self._writer is None:
            print("Initializing Summary Writer with log_dir={}".format(log_dir))
            self._writer = SummaryWriter(log_dir=log_dir)

    def close_writer(self):
        if self._writer is not None:
            self._writer.close()

    def write_scalar(self, tag, scalar_value, global_step=None):
        if self._writer is not None:
            self._writer.add_scalar(tag, scalar_value, global_step)

    def train(self):
        self.before_train()
        for self.current_epoch in range(self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            # exit()
            self.after_epoch()
        self.after_train()

    def before_train(self):
        self.time_start = time.time()

        # Initialise Weights & Biases (opt-in via WANDB.ENABLED in config)
        self._wandb_enabled = (
            WANDB_AVAILABLE
            and getattr(self.cfg, "WANDB", None) is not None
            and getattr(self.cfg.WANDB, "ENABLED", False)
        )
        if self._wandb_enabled:
            wandb_cfg = self.cfg.WANDB
            run_name = getattr(wandb_cfg, "RUN_NAME", None)
            wandb.init(
                project=getattr(wandb_cfg, "PROJECT", "swift"),
                name=run_name,
                group=getattr(wandb_cfg, "GROUP", None),
                config={
                    "model": self.cfg.MODEL.NAME,
                    "backbone": getattr(self.cfg.MODEL.Adapter, "BACKBONE", "unknown"),
                    "dataset": self.cfg.DATASET.NAME,
                    "source_domains": list(self.cfg.DATASET.SOURCE_DOMAINS),
                    "lr": self.cfg.OPTIM.LR,
                    "max_epoch": self.cfg.OPTIM.MAX_EPOCH,
                    "batch_size": self.cfg.DATALOADER.TRAIN.BATCH_SIZE,
                    "num_instances": self.cfg.DATALOADER.TRAIN.NUM_INSTANCES,
                    "weight_decay": self.cfg.OPTIM.WEIGHT_DECAY,
                    "warmup_epoch": self.cfg.OPTIM.WARMUP_EPOCH,
                    "warmup_type": self.cfg.OPTIM.WARMUP_TYPE,
                    "lr_scheduler": self.cfg.OPTIM.LR_SCHEDULER,
                    "triplet_loss_weight": self.cfg.MODEL.TRIPLET_LOSS_WEIGHT,
                    "margin": getattr(self.cfg.SOLVER, "MARGIN", None),
                    "day_night_adapter": self.cfg.MODEL.Day_Night_Adapter,
                    "seed": self.cfg.SEED,
                },
                dir=self.output_dir,
                reinit=True,
            )
            print(f"W&B run initialised: {wandb.run.url}")

    def after_train(self):
        print("Finish Training")
        results = self.test()
        if self._wandb_enabled and results:
            wandb.log({f"eval/{k}": v for k, v in results.items()})
            wandb.finish()

    def run_epoch(self):
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.data_loader_train)
        end_time = time.time()

        for self.batch_idx, batch_data in enumerate(self.data_loader_train):
            data_time.update(time.time() - end_time)
            # Delegate parsing to the helper so train/test share the same contract
            loss_summary = self.forward_backward(batch_data)

            batch_time.update(time.time() - end_time)
            losses.update(loss_summary)

            if self._should_print_train_status(
                self.batch_idx, self.num_batches, self.cfg.TRAIN.PRINT_FREQ
            ):
                num_batches_remain = 0
                num_batches_remain += self.num_batches - self.batch_idx - 1
                num_batches_remain += (
                    self.max_epoch - self.current_epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * num_batches_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.current_epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                # Include domain_id when each domain has separate LR
                if 'domain_id' in loss_summary and len(self.cfg.OPTIM.DOMAIN_OPTIM.DOMAIN_LR_MULTIPLIERS) > 1:
                    info += [f"domain_id {loss_summary['domain_id']}"]
                    info += [f"domain {loss_summary['domain']}"]    
                    active_model_name = f"dino_adapter_domain_{loss_summary['domain_id']}"
                    current_lr = self.get_current_lr(active_model_name)
                else:
                    current_lr = self.get_current_lr()
                info += [f"{losses}"]
                info += [f"lr {current_lr:.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

                # Log to W&B at each print step
                if self._wandb_enabled:
                    global_step = self.current_epoch * self.num_batches + self.batch_idx
                    log_dict = {f"train/{k}": meter.avg for k, meter in losses.meters.items()}
                    log_dict["train/lr"] = current_lr
                    log_dict["epoch"] = self.current_epoch + 1
                    wandb.log(log_dict, step=global_step)

            end_time = time.time()

    def before_epoch(self):
        pass

    @staticmethod
    def _should_print_train_status(batch_idx, num_batches, print_freq):
        if print_freq <= 0:
            return True

        is_first_batch = batch_idx == 0
        is_periodic_batch = (batch_idx + 1) % print_freq == 0
        is_last_batch = (batch_idx + 1) == num_batches

        return is_first_batch or is_periodic_batch or is_last_batch

    def after_epoch(self):
        # Step LR scheduler once per epoch
        self.update_lr()
        if self._wandb_enabled:
            # Log the new LR (post-step) for every registered domain optimizer
            lr_logs = {}
            for model_name, opt in self._optimizers.items():
                if opt is not None:
                    lr_logs[f"lr/{model_name}"] = opt.param_groups[0]["lr"]
            wandb.log(lr_logs, step=(self.current_epoch + 1) * self.num_batches)
            
        # Optional periodic evaluation
        if self.cfg.TEST.EVAL_PERIOD > 0:
            if (self.current_epoch + 1) % self.cfg.TEST.EVAL_PERIOD == 0 or (self.current_epoch + 1) == self.max_epoch:
                results = self.test()
                if self._wandb_enabled and results:
                    wandb.log({f"eval/{k}": v for k, v in results.items()}, step=(self.current_epoch + 1) * self.num_batches)
                self.set_model_mode("train")
                
        if self.current_epoch + 1 == self.max_epoch:
            self.save_model(self.current_epoch, self.output_dir)

    @torch.no_grad()
    def test(self, split=None):
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT
        if split == "Validation" and self.val_loader is not None:
            data_loader = self.data_loader_val
        elif split == "Test":
            data_loader = self.data_loader_test
        else:
            raise NotImplementedError

        print("Evaluate on the {} Set".format(split))

        for _, batch_data in enumerate(tqdm(data_loader)):
            input_data, target, domain, time, camids, img_paths = self.parse_batch_test(batch_data)

            output = self.model_inference(input_data, domain, time)
            self.evaluator.process(
                (output.cpu(), target.cpu().tolist(), camids, domain, img_paths)
            )
        results = self.evaluator.evaluate()

        return results

    def parse_batch_train(self, batch_data):
        image = batch_data["imgs"].to(self.device)
        target = batch_data["aids"].to(self.device)
        domain = batch_data["domains"]
        time = batch_data["time"]

        return image, target, domain, time

    def parse_batch_test(self, batch_data):
        input_data = batch_data["imgs"].to(self.device)
        target = batch_data["aids"].to(self.device)
        domain = batch_data["domains"]
        camids = batch_data["camids"]
        time = batch_data["time"]
        img_paths = batch_data["img_paths"]
        return input_data, target, domain, time, camids, img_paths

    def forward_backward(self, batch_data):
        raise NotImplementedError

    def get_current_lr(self, model_names=None):
        model_name = self.get_model_names(model_names)[0]
        return self._optimizers[model_name].param_groups[0]["lr"]

    def model_registeration(
        self, model_name="model", model=None, optimizer=None, lr_scheduler=None
    ):
        assert model_name not in self._models, "Found duplicate model names."

        self._models[model_name] = model
        self._optimizers[model_name] = optimizer
        self._lr_schedulers[model_name] = lr_scheduler

    def get_model_names(self, model_names=None):
        if model_names is not None:
            if not isinstance(model_names, list):
                model_names = [model_names]

            for model_name in model_names:
                assert model_name in list(self._models.keys())
            return model_names
        else:
            return list(self._models.keys())

    def model_inference(self, input_data, domain, time=None):
        _, feat = self.model(input_data)
        # Normalize features for evaluation stability
        feat = torch.nn.functional.normalize(feat, dim=-1, eps=1e-6)
        return feat
    
    def save_model(
        self,
        current_epoch,
        save_dir,
        model_name="name",
    ):
        model_names = self.get_model_names()

        for model_name in model_names:
            model_dict = self._models[model_name].state_dict()

            optimizer_dict = None
            if self._optimizers[model_name] is not None:
                optimizer_dict = self._optimizers[model_name].state_dict()

            lr_scheduler_state_dict = None
            if self._lr_schedulers[model_name] is not None:
                lr_scheduler_state_dict = self._lr_schedulers[model_name].state_dict()

            # Remove "module." in state_dict's keys
            new_model_dict = OrderedDict()
            for key, value in model_dict.items():
                if key.startswith("module."):
                    key = key[7:]
                new_model_dict[key] = value
            model_dict = new_model_dict

            model_domains = self.cfg.DATASET.TARGET_DOMAINS if hasattr(self.cfg.DATASET, 'TARGET_DOMAINS') else []
            if len(model_domains) > 0:
                model_domains = "-".join(model_domains)
            model_name = self.cfg.MODEL.NAME
            current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
            model_full_name = f"{model_name}_domains_{model_domains}_{current_time}.pth.tar"

            fpath = os.path.join(save_dir, model_full_name + str(current_epoch + 1))
            torch.save(
                {
                    "state_dict": model_dict,
                    "epoch": current_epoch + 1,
                    "optimizer": optimizer_dict,
                    "lr_scheduler": lr_scheduler_state_dict,
                },
                fpath,
            )

            print("Model Saved to: {}".format(fpath))
