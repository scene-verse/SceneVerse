import os
import json
import collections
from pathlib import Path

import numpy as np
import torch

from data.data_utils import SQA3DAnswer
from evaluator.build import EVALUATOR_REGISTRY


@EVALUATOR_REGISTRY.register()
class SQA3DEval():
    # 0: what, 1: is, 2: how, 3: can, 4: which, 5: others
    def __init__(self, cfg, task_name):
        self.eval_dict = {
            'target_metric': [], 'obj_cls_raw_acc': [],'ans1_acc': [], 'ans10_acc': [],
            'type0_acc': [], 'type1_acc': [], 'type2_acc': [],
            'type0_acc': [], 'type1_acc': [], 'type2_acc': [],
            'type3_acc': [], 'type4_acc': [], 'type5_acc': []
        }
        # run
        self.total_count = 0
        self.type_count = {
            'type0_count': 1e-10, 'type1_count': 1e-10, 'type2_count': 1e-10,
            'type3_count': 1e-10, 'type4_count': 1e-10, 'type5_count': 1e-10
        }
        self.best_result = -np.inf
        self.base_dir = cfg.data.scan_family_base

        answer_data = json.load(
            open(os.path.join(self.base_dir,
                              'annotations/sqa_task/answer_dict.json'), encoding='utf-8')
        )[0]
        answer_counter = []
        for data in answer_data.keys():
            answer_counter.append(data)
        answer_counter = collections.Counter(sorted(answer_counter))
        answer_cands = answer_counter.keys()
        self.answer_vocab = SQA3DAnswer(answer_cands)

        self.save = cfg.eval.save
        if self.save:
            self.eval_results = []
            self.save_dir = Path(cfg.exp_dir) / "eval_results" / task_name
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def update(self, data_dict):
        metrics = self.batch_metrics(data_dict)
        batch_count = metrics['total_count']
        self.total_count += batch_count
        for key in metrics:
            if 'type' in key and 'count' in key:
                self.type_count[key] += metrics[key]

        if self.save:
            for i in range(metrics["total_count"]):
                self.eval_results.append({
                    # vision
                    "source": data_dict['source'][i],
                    "scan_id": data_dict['scan_id'][i],
                    "anchor": data_dict['anchor_locs'][i],
                    'anchor_ort': data_dict['anchor_orientation'][i],
                    # language
                    "instruction": data_dict['prompt_after_obj'][i],
                    "response_gt": data_dict['answer_list'][i].split('[answer_seq]'),
                    "response_pred": data_dict['output_text'][i]
                })

        # save eval dict
        for key in self.eval_dict.keys():
            if 'type' in key:
                self.eval_dict[key].append(float(metrics[key]) * metrics['type' + key[4] + '_count'])
            else:
                self.eval_dict[key].append(float(metrics[key]) * batch_count)

    def batch_metrics(self, data_dict):
        metrics = {}

        # ans
        choice_1 = data_dict['answer_scores'].argmax(dim=-1)
        choice_10 = torch.topk(data_dict['answer_scores'].detach(), 10, -1)[1]
        correct1 = 0
        correct10 = 0
        correct_type = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        count_type = {0: 1e-10, 1: 1e-10, 2: 1e-10, 3: 1e-10, 4: 1e-10, 5: 1e-10}
        for i in range(data_dict['answer_label'].shape[0]):
            count_type[data_dict['sqa_type'][i].item()] += 1
            if data_dict['answer_label'][i, choice_1[i]] == 1:
                correct1 += 1
                correct_type[data_dict['sqa_type'][i].item()] += 1
            for j in range(10):
                if data_dict['answer_label'][i, choice_10[i, j]] == 1:
                    correct10 += 1
                    break
        metrics['ans1_acc'] = correct1 / float(len(choice_1))
        metrics['ans10_acc'] = correct10 / float(len(choice_1))
        # metrics['answer_top10'] = [
        #     # TODO: add this answer vocabulary in dataloader
        #     [self.answer_vocab.itos(choice_10[i, j].item()) for j in range(10)] for i in
        #     range(choice_10.shape[0])
        # ]

        metrics['obj_cls_raw_acc'] = torch.sum(
            torch.argmax(data_dict['obj_cls_raw_logits'], dim=2)[data_dict['obj_masks']] == data_dict["obj_labels"][
                data_dict['obj_masks']]).item() / float(data_dict['obj_masks'].sum().item())

        # question type acc
        for key in count_type.keys():
            metrics['type' + str(key) + '_acc'] = correct_type[key] / count_type[key]
            metrics['type' + str(key) + '_count'] = count_type[key]

        metrics['target_metric'] = metrics['ans1_acc']
        metrics["total_count"] = data_dict["answer_scores"].shape[0]
        return metrics

    def reset(self):
        for key in self.eval_dict.keys():
            self.eval_dict[key] = []
        self.total_count = 0
        self.type_count = {
            'type0_count': 1e-10, 'type1_count': 1e-10, 'type2_count': 1e-10, 
            'type3_count': 1e-10, 'type4_count': 1e-10, 'type5_count': 1e-10
        }
        if self.save:
            self.eval_results = []

    def record(self, split='val'):
        # record
        for k, v in self.eval_dict.items():
            if k == "answer_top10":
                continue
            if 'type' in k:
                self.eval_dict[k] = sum(v) / self.type_count['type' + k[4] + '_count']
            else:
                self.eval_dict[k] = sum(v) / self.total_count

        if self.eval_dict["target_metric"] > self.best_result:
            is_best = True
            self.best_result = self.eval_dict["target_metric"]
        else:
            is_best = False

        if self.save and (is_best or split == 'test'):
            torch.save(self.eval_results, str(self.save_dir / 'results.pt'))

        return is_best, self.eval_dict
