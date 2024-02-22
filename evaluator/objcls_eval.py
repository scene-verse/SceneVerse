import torch
from pathlib import Path

from evaluator.build import EVALUATOR_REGISTRY, BaseEvaluator


@EVALUATOR_REGISTRY.register()
class PretrainObjEval(BaseEvaluator):
    def __init__(self, cfg, accelerator, **kwargs):
        self.target_metric = "accuracy"
        self.save_dir = Path(cfg.exp_dir) / "eval_results" / self.__class__.__name__
        super().__init__(cfg, accelerator, **kwargs)

    def batch_metrics(self, data_dict, include_count=False):
        metrics = {}
        logits = data_dict["obj_logits"][data_dict["obj_masks"]].view(-1, data_dict["obj_logits"].shape[-1])
        labels = data_dict["obj_labels"][data_dict["obj_masks"]].view(-1)
        _, pred = torch.max(logits, 1)
        metrics["accuracy"] = ((pred == labels.view(-1)).sum().item(), labels.shape[0])
        if not include_count:
            for key, v in metrics.items():
                metrics[key] = v[0] / max(v[1], 1)
        return metrics