import copy
from tqdm import tqdm

import torch
from trainer.build import TRAINER_REGISTRY
from trainer.build import BaseTrainer


@TRAINER_REGISTRY.register()
class DefaultTrainer(BaseTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.best_metric = -1

    def forward(self, data_dict):
        return self.model(data_dict)

    def backward(self, loss):
        self.optimizer.zero_grad()
        self.accelerator.backward(loss)
        if self.grad_norm is not None and self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.grad_norm)
        self.optimizer.step()
        self.scheduler.step()

    def train_step(self, epoch):
        self.model.train()
        loader = self.data_loaders["train"]
        pbar = tqdm(range(len(loader)), disable=(not self.accelerator.is_main_process), desc=f"[Epoch {epoch + 1}/{self.epochs}]")
        for i, data_dict in enumerate(loader):
            with self.accelerator.accumulate(self.model):
                data_dict['cur_step'] = epoch * len(loader) + i
                data_dict['total_steps'] = self.total_steps
                # forward
                data_dict = self.forward(data_dict)
                # calculate loss
                loss, losses = self.loss(data_dict)
                # calculate evaluator
                metrics = self.evaluator.batch_metrics(data_dict)
                # optimize
                self.backward(loss)
                # record
                self.global_step += 1
                log_dict = {'step': self.global_step}
                log_dict.update(losses)
                log_dict.update(metrics)
                self.log(log_dict, mode="train")
                pbar.update(1)

    @torch.no_grad()
    def eval_step(self, epoch):
        self.model.eval()
        loader = self.data_loaders["val"]
        pbar = tqdm(range(len(loader)), disable=(not self.accelerator.is_main_process))
        for i, data_dict in enumerate(loader):
            data_dict = self.forward(data_dict)
            self.evaluator.update(data_dict)
            pbar.update(1)
        is_best, results = self.evaluator.record()
        if is_best:
            self.best_metric = results["target_metric"]
        self.log(results, mode="val")
        self.evaluator.reset()
        return is_best

    @torch.no_grad()
    def test_step(self):
        self.model.eval()
        loader = self.data_loaders["test"]
        pbar = tqdm(range(len(loader)), disable=(not self.accelerator.is_main_process))
        for i, data_dict in enumerate(loader):
            data_dict = self.forward(data_dict)
            self.evaluator.update(data_dict)
            pbar.update(1)
        is_best, results = self.evaluator.record()
        self.log(results, mode="test")
        self.evaluator.reset()
        return results

    def run(self):
        if self.mode == "train":
            start_epoch = self.exp_tracker.epoch
            self.global_step = start_epoch * len(self.data_loaders["train"])
            for epoch in range(start_epoch, self.epochs):
                self.exp_tracker.step()
                self.train_step(epoch)

                if self.epochs_per_eval and (epoch + 1) % self.epochs_per_eval == 0:
                    is_best = self.eval_step(epoch)
                    self.accelerator.print(f"[Epoch {epoch + 1}/{self.epochs}] finished eval, is_best: {is_best}")
                else:
                    is_best = False

                self.accelerator.wait_for_everyone()
                if self.accelerator.is_main_process:
                    self.save("latest.pth")
                    if is_best:
                        self.save("best.pth")
                    if self.epochs_per_save and (epoch + 1) % self.epochs_per_save == 0:
                        self.save(f"ckpt_{epoch+1}.pth")

        self.test_step()
        if self.mode == "train":
            self.accelerator.end_training()
