import copy as cp
import glob
from datetime import timedelta
from pathlib import Path
from omegaconf import OmegaConf
from omegaconf import open_dict
from tqdm import tqdm
import numpy as np

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed, InitProcessGroupKwargs
from fvcore.common.registry import Registry
import torch
import wandb

import common.io_utils as iu
from common.io_utils import make_dir
import common.misc as misc
from data.build import build_dataloader
from evaluator.build import build_eval
from model.build import build_model
from optim.build import build_optim


TRAINER_REGISTRY = Registry("Trainer")


class Tracker():
    def __init__(self, cfg):
        self.reset(cfg)

    def step(self):
        self.epoch += 1

    def reset(self, cfg):
        self.exp_name = f"{cfg.exp_dir.parent.name.replace(f'{cfg.name}', '').lstrip('_')}/{cfg.exp_dir.name}"
        self.epoch = 0
        self.best_result = -np.inf

    def state_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('__')}
    
    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

@TRAINER_REGISTRY.register()
class BaseTrainer():
    def __init__(self, cfg):
        set_seed(cfg.rng_seed)
        self.debug = cfg.debug.get("flag", False)
        self.hard_debug = cfg.debug.get("hard_debug", False)
        self.epochs_per_eval = cfg.solver.get("epochs_per_eval", None)
        self.epochs_per_save = cfg.solver.get("epochs_per_save", None)
        self.global_step = 0
        
        # Initialize accelerator
        self.exp_tracker = Tracker(cfg)
        wandb_args = {"entity": cfg.logger.entity, "id": cfg.logger.run_id, "resume": cfg.resume}
        if not cfg.logger.get('autoname'):
            wandb_args["name"] = self.exp_tracker.exp_name
        # There is bug in logger setting, needs fixing from accelerate side
        self.logger = get_logger(__name__)
        self.mode = cfg.mode

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))
        kwargs = ([ddp_kwargs] if cfg.num_gpu > 1 else []) + [init_kwargs]

        gradient_accumulation_steps = cfg.solver.get("gradient_accumulation_steps", 1)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            log_with=cfg.logger.name,
            kwargs_handlers=kwargs
        )
        if not self.hard_debug:
            self.accelerator.init_trackers(
                    project_name=cfg.name if not self.debug else "Debug",
                    config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True) if not cfg.resume else None,
                    init_kwargs={"wandb": wandb_args}
                )
        print(OmegaConf.to_yaml(cfg))

        if cfg.model.name == 'Query3D':
            # choose whether to load mv or voxel features based on model.memories for Query3D
            # TODO: a better way to do this?
            if 'mv' in cfg.model.memories or 'sem' in cfg.model.memories:
                cfg.data.load_multiview_info = True
            if 'voxel' in cfg.model.memories or 'sem' in cfg.model.memories:
                cfg.data.load_mask3d_voxel = True
            txt_model2tokenizer = {'BERTLanguageEncoder': 'bert-base-uncased', 'CLIPLanguageEncoder': 'openai/clip-vit-large-patch14'}
            cfg.data_wrapper.tokenizer = txt_model2tokenizer[cfg.model.txt_encoder.name]
            
        keys = ["train", "val", "test"] if self.mode == "train" else ["test"]
        self.data_loaders = {key : build_dataloader(cfg, split=key) for key in keys}
        self.model = build_model(cfg)
        if self.mode == "test":
            total_steps = 1
        else:
            total_steps = (len(self.data_loaders["train"]) * cfg.solver.epochs) // gradient_accumulation_steps
        self.loss, self.optimizer, self.scheduler = build_optim(cfg, self.model.get_opt_params(),
                                                                total_steps= total_steps)

        if misc.rgetattr(cfg, "eval.pass_kwargs", False):
            kwargs = {"dataloaders": self.data_loaders}
        else:
            kwargs = {}
        self.evaluator = build_eval(cfg, self.accelerator, **kwargs)

        # Training details
        self.epochs = cfg.solver.epochs
        self.total_steps = 1 if self.mode == "test" else len(self.data_loaders["train"]) * cfg.solver.epochs
        self.grad_norm = cfg.solver.get("grad_norm")

        # Load pretrain model weights
        if cfg.get('pretrain_ckpt_path'):
            self.pretrain_ckpt_path = Path(cfg.pretrain_ckpt_path)
            self.load_pretrain()

        # Accelerator preparation
        self.model, self.loss, self.optimizer, self.scheduler = self.accelerator.prepare(self.model, self.loss, self.optimizer, self.scheduler)
        for name, loader in self.data_loaders.items():
            if isinstance(loader, list):
                loader = self.accelerator.prepare(*loader)
            else:
                loader = self.accelerator.prepare(loader)
            self.data_loaders[name] = loader
        self.accelerator.register_for_checkpointing(self.exp_tracker)

        # Check if resuming from previous checkpoint is needed
        self.ckpt_path = Path(cfg.ckpt_path) if cfg.get("ckpt_path") else Path(cfg.exp_dir) / "ckpt" / "best.pth"
        if cfg.resume:
            self.resume()

    def forward(self, data_dict):
        return self.model(data_dict)

    def backward(self, loss):
        # Need to be reimplemented when using different sets of optimizer and schedulers
        self.optimizer.zero_grad()
        self.accelerator.backward(loss)
        if self.grad_norm is not None and self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.grad_norm)
        self.optimizer.step()
        self.scheduler.step()

    def log(self, results, mode="train"):
        if not self.hard_debug:
            log_dict = {}
            for key, val in results.items():
                if isinstance(val, torch.Tensor):
                    val = val.item()
                log_dict[f"{mode}/{key}"] = val
            if mode == "train":
                lrs = self.scheduler.get_lr()
                for i, lr in enumerate(lrs):
                    log_dict[f"{mode}/lr/group_{i}"] = lr
            self.accelerator.log(log_dict, step=self.global_step)

    def save(self, name):
        make_dir(self.ckpt_path.parent)
        self.save_func(str(self.ckpt_path.parent / name))

    def resume(self):
        if self.ckpt_path.exists():
            print(f"Resuming from {str(self.ckpt_path)}")
            # self.logger.info(f"Resuming from {str(self.ckpt_path)}")
            self.accelerator.load_state(str(self.ckpt_path))
            # self.logger.info(f"Successfully resumed from {self.ckpt_path}")
            print(f"Successfully resumed from {self.ckpt_path}")
        else:
            self.logger.info("training from scratch")

    def load_pretrain(self):
        self.logger.info(f"Loading pretrained weights from {str(self.pretrain_ckpt_path)}")
        model_weight_path_pattern = str(self.pretrain_ckpt_path / "pytorch_model*.bin")
        model_weight_paths = glob.glob(model_weight_path_pattern)
        if len(model_weight_paths) == 0:
            raise FileNotFoundError(f"Cannot find pytorch_model.bin in {str(self.pretrain_ckpt_path)}")
        weights = {}
        for model_weight_path in model_weight_paths:
            weights.update(torch.load(model_weight_path, map_location="cpu"))
        warning = self.model.load_state_dict(weights, strict=False)
        self.logger.info(f"Successfully loaded from {str(self.pretrain_ckpt_path)}: {warning}")

    def save_func(self, path):
        self.accelerator.save_state(path)
    

def build_trainer(cfg):
    return TRAINER_REGISTRY.get(cfg.trainer)(cfg)
