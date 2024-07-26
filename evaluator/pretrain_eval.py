import torch
import numpy as np

from evaluator.build import EVALUATOR_REGISTRY, BaseEvaluator


@EVALUATOR_REGISTRY.register()
class PretrainEval(BaseEvaluator):
    def __init__(self, cfg, accelerator, **kwargs):
        self.cfg = cfg
        self.eval_dict = {
            "target_metric": [], "og_acc": [], "lang_cls_acc_mask": [], "obj_cls_post_acc": [], "obj_cls_pre_acc": [],
            "obj_cls_raw_acc": [], "obj_cls_pre_acc_unmask": [], "obj_cls_pre_acc_mask": [],
            "obj_cls_post_acc_unmask": [], "obj_cls_post_acc_mask": []
        }
        self.accelerator = accelerator
        self.device = self.accelerator.device
        self.total_count = 0
        self.best_result = -np.inf

    def batch_metrics(self, data_dict):
        metrics = {}
        txt_token_mask = (data_dict['masked_lm_labels'] != -1)
        if 'tgt_object_id' in data_dict.keys():
            metrics['og_acc'] = (torch.argmax(data_dict['og3d_logits'], dim=-1) == data_dict['tgt_object_id'].squeeze(
                1)).sum().item() / float(len(data_dict['tgt_object_id']))
        metrics['lang_cls_acc_mask'] = torch.sum(
            torch.argmax(data_dict['txt_lm_cls_logits'], dim=2)[txt_token_mask] == data_dict['masked_lm_labels'][
                txt_token_mask]).item() / float(txt_token_mask.sum().item() + 1e-8)
        if 'obj_cls_post_logits' in data_dict.keys():
            metrics['obj_cls_post_acc'] = torch.sum(
                torch.argmax(data_dict['obj_cls_post_logits'], dim=2)[data_dict['obj_masks']] == data_dict["obj_labels"][
                    data_dict['obj_masks']]).item() / float(data_dict['obj_masks'].sum().item() + 1e-8)
            metrics['obj_cls_post_acc_unmask'] = torch.sum(
                torch.argmax(data_dict['obj_cls_post_logits'], dim=2)[
                    data_dict['obj_masks'] * data_dict['obj_sem_masks']] ==
                data_dict["obj_labels"][data_dict['obj_masks'] * data_dict['obj_sem_masks']]).item() / float(
                (data_dict['obj_masks'] * data_dict['obj_sem_masks']).sum().item() + 1e-8)
            metrics['obj_cls_post_acc_mask'] = torch.sum(torch.argmax(data_dict['obj_cls_post_logits'], dim=2)[
                                                             data_dict['obj_masks'] * data_dict[
                                                                 'obj_sem_masks'].logical_not()] ==
                                                         data_dict["obj_labels"][
                                                             data_dict['obj_masks'] * data_dict[
                                                                 'obj_sem_masks'].logical_not()]).item() / float(
                (data_dict['obj_masks'] * data_dict['obj_sem_masks'].logical_not()).sum().item() + 1e-8)
        if 'obj_cls_raw_logits' in data_dict.keys():
            metrics['obj_cls_raw_acc'] = torch.sum(
                torch.argmax(data_dict['obj_cls_raw_logits'], dim=2)[data_dict['obj_masks']] == data_dict["obj_labels"][
                    data_dict['obj_masks']]).item() / float(data_dict['obj_masks'].sum().item() + 1e-8)
        if 'obj_cls_pre_logits' in data_dict.keys():
            metrics['obj_cls_pre_acc'] = torch.sum(
                torch.argmax(data_dict['obj_cls_pre_logits'], dim=2)[data_dict['obj_masks']] == data_dict["obj_labels"][
                    data_dict['obj_masks']]).item() / float(data_dict['obj_masks'].sum().item() + 1e-8)
            metrics['obj_cls_pre_acc_unmask'] = torch.sum(
                torch.argmax(data_dict['obj_cls_pre_logits'], dim=2)[data_dict['obj_masks'] * data_dict['obj_sem_masks']] ==
                data_dict["obj_labels"][data_dict['obj_masks'] * data_dict['obj_sem_masks']]).item() / float(
                (data_dict['obj_masks'] * data_dict['obj_sem_masks']).sum().item() + 1e-8)
            metrics['obj_cls_pre_acc_mask'] = torch.sum(torch.argmax(data_dict['obj_cls_pre_logits'], dim=2)[
                                                            data_dict['obj_masks'] * data_dict[
                                                                'obj_sem_masks'].logical_not()] == data_dict["obj_labels"][
                                                            data_dict['obj_masks'] * data_dict[
                                                                'obj_sem_masks'].logical_not()]).item() / float(
                (data_dict['obj_masks'] * data_dict['obj_sem_masks'].logical_not()).sum().item() + 1e-8)
        all_acc = [v for k, v in metrics.items()]
        metrics["target_metric"] = float(sum(all_acc)) / len(all_acc)
        metrics["total_count"] = data_dict["txt_lm_cls_logits"].shape[0]
        return metrics

    def update(self, data_dict):
        metrics = self.batch_metrics(data_dict)
        self.total_count += metrics["total_count"]
        for key in self.eval_dict.keys():
            if key not in metrics.keys():
                continue
            self.eval_dict[key].append(float(metrics[key]) * metrics["total_count"])

    def record(self):
        # Average
        for k, v in self.eval_dict.items():
            self.eval_dict[k] = sum(v) / self.total_count
        if self.eval_dict["target_metric"] > self.best_result:
            is_best = True
            self.best_result = self.eval_dict["target_metric"]
        else:
            is_best = False
        return is_best, self.eval_dict

    def reset(self):
        for key in self.eval_dict.keys():
            self.eval_dict[key] = []
        self.total_count = 0