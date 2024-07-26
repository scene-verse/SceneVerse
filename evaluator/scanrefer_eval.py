from pathlib import Path
import torch

from evaluator.build import EVALUATOR_REGISTRY, BaseEvaluator


@EVALUATOR_REGISTRY.register()
class ScanReferEval(BaseEvaluator):
    def __init__(self, cfg, accelerator, **kwargs):
        self.target_metric = 'og_acc_iou25'
        self.save_dir = Path(cfg.exp_dir) / "eval_results" / self.__class__.__name__
        super().__init__(cfg, accelerator, **kwargs)

    def batch_metrics(self, data_dict, include_count=False):
        # Per-scene eval
        if len(data_dict['tgt_object_id_iou25'].shape) == 3:
            data_dict['tgt_object_id_iou25'] = data_dict['tgt_object_id_iou25'].flatten(0, 1)
            data_dict['tgt_object_id_iou50'] = data_dict['tgt_object_id_iou50'].flatten(0, 1)
            data_dict['tgt_object_id'] = data_dict['tgt_object_id'].flatten(0, 1).unsqueeze(1)
            data_dict['is_multiple'] = data_dict['is_multiple'].flatten(0, 1)
            data_dict['og3d_logits'] = data_dict['og3d_logits'].flatten(0, 1)

        metrics = {}
        og_pred = torch.argmax(data_dict['og3d_logits'], dim=-1)
        total_count = len(og_pred)

        multiple_count = data_dict['is_multiple'].sum().item()
        unique_count = total_count - multiple_count

        # Correct counts for iou25 and iou50
        iou25_correct_mask = data_dict['tgt_object_id_iou25'][torch.arange(len(og_pred)), og_pred].to(bool)
        iou50_correct_mask = data_dict['tgt_object_id_iou50'][torch.arange(len(og_pred)), og_pred].to(bool)
        iou25_correct = iou25_correct_mask.sum().item()
        iou50_correct = iou50_correct_mask.sum().item()

        # Correct counts for unique and multiple iou25 and iou50
        iou25_multiple_correct = (iou25_correct_mask & data_dict['is_multiple']).sum().item()
        iou25_unique_correct = iou25_correct - iou25_multiple_correct

        iou50_multiple_correct = (iou50_correct_mask & data_dict['is_multiple']).sum().item()
        iou50_unique_correct = iou50_correct - iou50_multiple_correct

        metrics['og_acc_iou25'] = iou25_correct
        metrics['og_acc_iou50'] = iou50_correct
        metrics['og_acc_iou25_unique'] = iou25_unique_correct
        metrics['og_acc_iou50_unique'] = iou50_unique_correct
        metrics['og_acc_iou25_multiple'] = iou25_multiple_correct
        metrics['og_acc_iou50_multiple'] = iou50_multiple_correct

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
                metrics[new_key] = ((pred[mask] == gt[mask]).sum().item(), mask.sum().item())

        for key in metrics:
            if isinstance(metrics[key], tuple):
                # already has count
                continue
            if 'unique' in key:
                metrics[key] = (metrics[key], unique_count)
            elif 'multiple' in key:
                metrics[key] = (metrics[key], multiple_count)
            else:
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
