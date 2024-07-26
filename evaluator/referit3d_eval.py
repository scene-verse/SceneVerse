from pathlib import Path
import torch

from evaluator.build import EVALUATOR_REGISTRY, BaseEvaluator


@EVALUATOR_REGISTRY.register()
class ReferIt3DEval(BaseEvaluator):
    def __init__(self, cfg, accelerator, **kwargs):
        self.target_metric = 'og_acc'
        self.save_dir = Path(cfg.exp_dir) / "eval_results" / self.__class__.__name__
        super().__init__(cfg, accelerator, **kwargs)

    def batch_metrics(self, data_dict, include_count=False):
        # Per-scene eval
        if len(data_dict['og3d_logits'].shape) == 3:
            data_dict['tgt_object_id'] = data_dict['tgt_object_id'].flatten(0, 1).unsqueeze(1)
            data_dict['is_hard'] = data_dict['is_hard'].flatten(0, 1)
            data_dict['is_view_dependent'] = data_dict['is_view_dependent'].flatten(0, 1)
            data_dict['og3d_logits'] = data_dict['og3d_logits'].flatten(0, 1)

        metrics = {}
        og_pred = torch.argmax(data_dict['og3d_logits'], dim=-1)
        total_count = len(og_pred)

        # Easy and hard counts
        hard_count = data_dict['is_hard'].sum().item()
        easy_count = total_count - hard_count

        # View-dependent and view-independent counts
        view_dep_count = data_dict['is_view_dependent'].sum().item()
        view_indep_count = total_count - view_dep_count

        # Correct counts
        correct_preds = data_dict['tgt_object_id'].flatten() == og_pred
        correct = correct_preds.sum().item()

        # Correct counts for easy and hard
        hard_correct = (correct_preds & data_dict['is_hard']).sum().item()
        easy_correct = correct - hard_correct

        # Correct counts for view-dependent and view-independent
        view_dep_correct = (correct_preds & data_dict['is_view_dependent']).sum().item()
        view_indep_correct = correct - view_dep_correct

        metrics['og_acc_easy'] = (easy_correct, easy_count)
        metrics['og_acc_hard'] = (hard_correct, hard_count)
        metrics['og_acc_view_dep'] = (view_dep_correct, view_dep_count)
        metrics['og_acc_view_indep'] = (view_indep_correct, view_indep_count)

        metrics['og_acc'] = (og_pred == data_dict['tgt_object_id'].squeeze(1)).sum().item()
        if 'txt_cls_logits' in data_dict:
            metrics['txt_acc'] = (torch.argmax(data_dict['txt_cls_logits'], dim=1) == data_dict["tgt_object_label"].squeeze(1)).sum().item() 
        
        # get obj cls acc
        gt = data_dict['obj_labels']
        mask = data_dict['obj_masks']
        for key in data_dict.keys():
            if key.endswith('logits') and data_dict[key].ndim == 3 and data_dict[key].shape[:2] == data_dict['obj_labels'].shape:
                new_key = key.replace('logits', 'acc')
                pred = torch.argmax(data_dict[key], dim=2)
                metrics[new_key] = ((pred[mask] == gt[mask]).sum().item(), data_dict['obj_masks'].sum().item())

        for key in metrics:
            if isinstance(metrics[key], tuple):
                # already has count
                continue
            metrics[key] = (metrics[key], total_count)
        
        if self.save:
            item_ids = data_dict['data_idx']
            for i in range(len(item_ids)):
                self.eval_results.append({
                    "scene_id": item_ids[i],
                    "bbox": data_dict['obj_boxes'][i][og_pred[i]].cpu().numpy().tolist(),
                    "correct": og_pred[i].item() == data_dict['tgt_object_id'][i].item()
                })

        if not include_count:
            for key, v in metrics.items():
                metrics[key] = v[0] / max(v[1], 1)

        return metrics
