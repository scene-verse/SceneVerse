import json
import numpy as np
from omegaconf import open_dict
from fvcore.common.registry import Registry

from common.misc import gather_dict

EVALUATOR_REGISTRY = Registry("EVALUATOR")


class BaseEvaluator():
    def __init__(self, cfg, accelerator):
        self.accelerator = accelerator
        self.best_result = -np.inf
        self.save = cfg.eval.save
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.reset()

    def reset(self):
        self.eval_results = []
        self.eval_dict = {}

    def batch_metrics(self, data_dict, include_count=False):
        raise NotImplementedError("Per batch metrics calculation is required for evaluation")

    def update(self, data_dict):
        metrics = self.batch_metrics(data_dict, include_count=True)
        for key in metrics.keys():
            if key not in self.eval_dict:
                self.eval_dict[key] = []
            self.eval_dict[key].append(metrics[key])

    def record(self):
        self.eval_dict = gather_dict(self.accelerator, self.eval_dict)
        for k, metrics in self.eval_dict.items():
            if not isinstance(metrics, list):
                continue
            # metrics is a list of (value, count)
            total_value = sum(x[0] for x in metrics)
            total_count = sum(x[1] for x in metrics)
            self.eval_dict[k] = total_value / max(total_count, 1)

        if self.save and self.accelerator.is_main_process:
            with (self.save_dir / "results.json").open("w") as f:
                json.dump(self.eval_results, f)
        
        self.eval_dict['target_metric'] = self.eval_dict[self.target_metric]
        if self.eval_dict["target_metric"] > self.best_result:
            is_best = True
            self.best_result = self.eval_dict["target_metric"]
        else:
            is_best = False
        self.eval_dict['best_result'] = self.best_result
        return is_best, self.eval_dict


def get_eval(name, cfg, accelerator, **kwargs):
    """Get an evaluator or a list of evaluators."""
    if isinstance(name, str):
        eval = EVALUATOR_REGISTRY.get(name)(cfg, accelerator, **kwargs)
    else:
        eval = [EVALUATOR_REGISTRY.get(i)(cfg, accelerator, **kwargs) for i in name]
    return eval

def build_eval(cfg, accelerator, **kwargs):
    if cfg.eval.get("train", None) is not None:
        train_eval = get_eval(cfg.eval.train.name, cfg, accelerator, **kwargs)
        val_eval = get_eval(cfg.eval.val.name, cfg, accelerator, **kwargs)
        return {"train": train_eval, "val": val_eval}
    elif cfg.eval.get("name", None) is not None:
        return get_eval(cfg.eval.name, cfg, accelerator, **kwargs)
    else:
        with open_dict(cfg):
            cfg.eval.name = [cfg.data.get(dataset).evaluator for dataset in cfg.data.val]
        return get_eval(cfg.eval.name, cfg, accelerator, **kwargs)