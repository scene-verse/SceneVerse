import os
import collections
import json
import random

import jsonlines
from tqdm import tqdm
import numpy as np
import albumentations as A
import volumentations as V
import torch
from torch.utils.data import Dataset
from pathlib import Path
from copy import deepcopy

from ..build import DATASET_REGISTRY
from ..data_utils import convert_pc_to_box, ScanQAAnswer, SQA3DAnswer, construct_bbox_corners, \
                         eval_ref_one_sample, is_explicitly_view_dependent, get_sqa_question_type
from .scannet_base import ScanNetBase


@DATASET_REGISTRY.register()
class ScanNetSQA3D(ScanNetBase):
    r"""
    questions json file: dict_keys(['info', 'license', 'data_type', 'data_subtype', 'task_type', 'questions'])
        'questions': List
        'questions'[0]: {
            'scene_id': 'scene0050_00',
            'situation': 'I am standing by the ottoman on my right facing a couple of toolboxes.',
            'alternative_situation': [
                'I just placed two backpacks on the ottoman on my right side before I went to play the piano in front of me to the right.',
                'I stood up from the ottoman and walked over to the piano ahead of me.'
            ],
            'question': 'What instrument in front of me is ebony and ivory?',
            'question_id': 220602000002
        }

    annotations json file: dict_keys(['info', 'license', 'data_type', 'data_subtype', 'annotations'])
        'annotations': List
        'annotations'[0]: {
            'scene_id': 'scene0050_00',
            'question_type': 'N/A',
            'answer_type': 'other',
            'question_id': 220602000002,
            'answers': [{'answer': 'piano', 'answer_confidence': 'yes', 'answer_id': 1}],
            'rotation': {'_x': 0, '_y': 0, '_z': -0.9995736030415032, '_w': -0.02919952230128897},
            'position': {'x': 0.7110268899979686, 'y': -0.03219739162793617, 'z': 0}
        }
    """
    def __init__(self, cfg, split):
        super().__init__(cfg, split)

        self.pc_type = cfg.data.args.pc_type
        self.sem_type = cfg.data.args.sem_type
        self.max_obj_len = cfg.data.args.max_obj_len - 1
        self.num_points = cfg.data.args.num_points
        self.filter_lang = cfg.data.args.filter_lang
        self.rot_aug = cfg.data.args.rot_aug
        self.use_unanswer = cfg.data.get(self.__class__.__name__).get(split).use_unanswer

        assert self.pc_type in ['gt', 'pred']
        assert self.sem_type in ['607']
        assert self.split in ['train', 'val', 'test']
        if self.split == 'train':
            self.pc_type = 'gt'
        # use test set for validation
        elif self.split == 'val':
            self.split = 'test'

        print(f"Loading ScanNet SQA3D {split}-set language")
        # build answer
        self.num_answers, self.answer_vocab, self.answer_cands = self.build_answer()

        # load annotations
        lang_data, self.scan_ids, self.scan_to_item_idxs = self._load_lang()
        if cfg.debug.flag:
            self.lang_data = []
            self.scan_ids = sorted(list(self.scan_ids))[:cfg.debug.debug_size]
            for item in lang_data:
                if item['scene_id'] in self.scan_ids:
                    self.lang_data.append(item)
        else:
            self.lang_data = lang_data

        # load question engine
        self.questions_map = self._load_question()
        print(f"Finish loading ScanNet SQA3D {split}-set language")

        # load scans
        print(f"Loading ScanNet SQA3D {split}-set scans")
        self.scan_data = self._load_scannet(self.scan_ids, self.pc_type, self.pc_type == 'gt')
        print(f"Finish loading ScanNet SQA3D {split}-set data")

    def __getitem__(self, index):
        item = self.lang_data[index]
        item_id = item['question_id']
        scan_id = item['scene_id']

        tgt_object_id_list = []
        tgt_object_name_list = []
        answer_list = [answer['answer'] for answer in item['answers']]
        answer_id_list = [self.answer_vocab.stoi(answer)
                          for answer in answer_list if self.answer_vocab.stoi(answer) >= 0]

        if self.split == 'train':
            # augment with random situation for train
            situation = random.choice(self.questions_map[scan_id][item_id]['situation'])
        else:
            # fix for eval
            situation = self.questions_map[scan_id][item_id]['situation'][0]
        question = self.questions_map[scan_id][item_id]['question']
        concat_sentence = situation + question
        question_type = get_sqa_question_type(question)

        # load pcds and labels
        if self.pc_type == 'gt':
            obj_pcds = self.scan_data[scan_id]['obj_pcds'] # N, 6
            obj_labels = self.scan_data[scan_id]['inst_labels'] # N
        elif self.pc_type == 'pred':
            obj_pcds = self.scan_data[scan_id]['obj_pcds_pred']
            obj_labels = self.scan_data[scan_id]['inst_labels_pred']

        # filter out background or language
        if self.filter_lang:
            if self.pc_type == 'gt':
                selected_obj_idxs = [i for i, obj_label in enumerate(obj_labels)
                                if (self.int2cat[obj_label] not in ['wall', 'floor', 'ceiling'])
                                and (self.int2cat[obj_label] in concat_sentence)]
                for _id in tgt_object_id_list:
                    if _id not in selected_obj_idxs:
                        selected_obj_idxs.append(_id)
            else:
                selected_obj_idxs = [i for i in range(len(obj_pcds))]
        else:
            if self.pc_type == 'gt':
                selected_obj_idxs = [i for i, obj_label in enumerate(obj_labels)
                                if (self.int2cat[obj_label] not in ['wall', 'floor', 'ceiling'])]
            else:
                selected_obj_idxs = [i for i in range(len(obj_pcds))]

        obj_pcds = [obj_pcds[id] for id in selected_obj_idxs]
        obj_labels = [obj_labels[id] for id in selected_obj_idxs]

        # build tgt object id and box
        if self.pc_type == 'gt':
            tgt_object_id_list = [selected_obj_idxs.index(x) for x in tgt_object_id_list]
            tgt_object_label_list = [obj_labels[x] for x in tgt_object_id_list]
            for i in range(len(tgt_object_label_list)):
                assert self.int2cat[tgt_object_label_list[i]] == tgt_object_name_list[i]
        elif self.pc_type == 'pred':
            # build gt box
            gt_center = []
            gt_box_size = []
            for cur_id in tgt_object_id_list:
                gt_pcd = self.scan_data[scan_id]["obj_pcds"][cur_id]
                center, box_size = convert_pc_to_box(gt_pcd)
                gt_center.append(center)
                gt_box_size.append(box_size)

            # start filtering
            tgt_object_id_list = []
            tgt_object_label_list = []
            for i in range(len(obj_pcds)):
                obj_center, obj_box_size = convert_pc_to_box(obj_pcds[i])
                for j in range(len(gt_center)):
                    if eval_ref_one_sample(construct_bbox_corners(obj_center, obj_box_size), construct_bbox_corners(gt_center[j], gt_box_size[j])) >= 0.25:
                        tgt_object_id_list.append(i)
                        tgt_object_label_list.append(self.cat2int[tgt_object_name_list[j]])
                        break
        assert(len(obj_pcds) == len(obj_labels))

        # crop objects
        if self.max_obj_len < len(obj_labels):
            selected_obj_idxs = tgt_object_id_list.copy()
            remained_obj_idx = []
            for kobj, klabel in enumerate(obj_labels):
                if kobj not in  tgt_object_id_list:
                    if klabel in tgt_object_label_list:
                        selected_obj_idxs.append(kobj)
                    else:
                        remained_obj_idx.append(kobj)
                if len(selected_obj_idxs) == self.max_obj_len:
                    break
            if len(selected_obj_idxs) < self.max_obj_len:
                random.shuffle(remained_obj_idx)
                selected_obj_idxs += remained_obj_idx[:(self.max_obj_len - len(selected_obj_idxs))]
            obj_pcds = [obj_pcds[i] for i in selected_obj_idxs]
            obj_labels = [obj_labels[i] for i in selected_obj_idxs]
            tgt_object_id_list = [i for i in range(len(tgt_object_id_list))]
            assert len(obj_pcds) == self.max_obj_len

        # rebuild tgt_object_id
        if len(tgt_object_id_list) == 0:
            tgt_object_id_list.append(len(obj_pcds))
            tgt_object_label_list.append(5)

        obj_fts, obj_locs, obj_boxes, obj_labels = self._obj_processing_post(obj_pcds, obj_labels, is_need_bbox=True, rot_aug=self.rot_aug)

        # convert answer format
        answer_label = torch.zeros(self.num_answers).long()
        for _id in answer_id_list:
            answer_label[_id] = 1
        # tgt object id
        tgt_object_id = torch.zeros(len(obj_fts) + 1).long() # add 1 for pad place holder
        for _id in tgt_object_id_list:
            tgt_object_id[_id] = 1
        # tgt object sematic
        if self.sem_type == '607':
            tgt_object_label = torch.zeros(607).long()
        else:
            raise NotImplementedError("semantic type " + self.sem_type)
        for _id in tgt_object_label_list:
            tgt_object_label[_id] = 1

        data_dict = {
            "situation": situation,
            "situation_pos": item['position'],
            "situation_rot": item['rotation'],
            "question": question,
            "sentence": concat_sentence,
            "scan_dir": os.path.join(self.base_dir, 'scans'),
            "scan_id": scan_id,
            "answer": "[answer_seq]".join(answer_list),
            "answer_label": answer_label, # A
            "tgt_object_id": torch.LongTensor(tgt_object_id), # N
            "tgt_object_label": torch.LongTensor(tgt_object_label), # L
            "obj_fts": obj_fts,
            "obj_locs": obj_locs,
            "obj_labels": obj_labels,
            "obj_boxes": obj_boxes, # N, 6
            "data_idx": item_id,
            "sqa_type": question_type
        }

        return data_dict

    def build_answer(self):
        answer_data = json.load(
            open(os.path.join(self.base_dir,
                              'annotations/sqa_task/answer_dict.json'), encoding='utf-8')
            )[0]
        answer_counter = []
        for data in answer_data.keys():
            answer_counter.append(data)
        answer_counter = collections.Counter(sorted(answer_counter))
        num_answers = len(answer_counter)
        answer_cands = answer_counter.keys()
        answer_vocab = SQA3DAnswer(answer_cands)
        print(f"total answers is {num_answers}")
        return num_answers, answer_vocab, answer_cands

    def _load_lang(self):
        lang_data = []
        scan_ids = set()
        scan_to_item_idxs = collections.defaultdict(list)

        anno_file = os.path.join(self.base_dir,
            f'annotations/sqa_task/balanced/v1_balanced_sqa_annotations_{self.split}_scannetv2.json')
        json_data = json.load(open(anno_file, 'r', encoding='utf-8'))['annotations']
        for item in json_data:
            if self.use_unanswer or (len(set(item['answers']) & set(self.answer_cands)) > 0):
                scan_ids.add(item['scene_id'])
                scan_to_item_idxs[item['scene_id']].append(len(lang_data))
                lang_data.append(item)
        print(f'{self.split} unanswerable question {len(json_data) - len(lang_data)},'
              + f'answerable question {len(lang_data)}')

        return lang_data, scan_ids, scan_to_item_idxs

    def _load_question(self):
        questions_map = {}
        anno_file = os.path.join(self.base_dir,
            f'annotations/sqa_task/balanced/v1_balanced_questions_{self.split}_scannetv2.json')
        json_data = json.load(open(anno_file, 'r', encoding='utf-8'))['questions']
        for item in json_data:
            if item['scene_id'] not in questions_map.keys():
                questions_map[item['scene_id']] = {}
            questions_map[item['scene_id']][item['question_id']] = {
                'situation': [item['situation']] + item['alternative_situation'],   # list of sentences
                'question': item['question']   # sentence
            }

        return questions_map


@DATASET_REGISTRY.register()
class ScanNetScanQAOld(ScanNetBase):
    def __init__(self, cfg, split):
        super(ScanNetScanQAOld, self).__init__(cfg, split)

        self.pc_type = cfg.data.args.pc_type
        self.sem_type = cfg.data.args.sem_type
        self.max_obj_len = cfg.data.args.max_obj_len - 1
        self.num_points = cfg.data.args.num_points
        self.filter_lang = cfg.data.args.filter_lang
        self.use_unanswer = cfg.data.get(self.__class__.__name__).get(split).use_unanswer

        assert self.pc_type in ['gt', 'pred']
        assert self.sem_type in ['607']
        assert self.split in ['train', 'val', 'test']
        if self.split == 'train':
            self.pc_type = 'gt'
        # TODO: hack test split to be the same as val
        if self.split == 'test':
            self.split = cfg.data.ScanNetScanQAOld.test.test_file
        self.is_test = ('test' in self.split)

        print(f"Loading ScanNet ScanQA {split}-set language")
        self.num_answers, self.answer_vocab, self.answer_cands = self.build_answer()
        lang_data, self.scan_ids, self.scan_to_item_idxs = self._load_lang()
        if cfg.debug.flag and cfg.debug.debug_size != -1:
            self.lang_data = []
            self.scan_ids = sorted(list(self.scan_ids))[:cfg.debug.debug_size]
            for item in lang_data:
                if item['scene_id'] in self.scan_ids:
                    self.lang_data.append(item)
        else:
            self.lang_data = lang_data
        print(f"Finish loading ScanNet ScanQA {split}-set language")

        print(f"Loading ScanNet ScanQA {split}-set scans")
        self.scan_data = self._load_scannet(self.scan_ids, self.pc_type,
                                           load_inst_info=('test' not in self.split))
        print(f"Finish loading ScanNet ScanQA {split}-set data")

    def __getitem__(self, index):
        """Data dict post-processing, for example, filtering, crop, nomalization,
        rotation, etc.
        Args:
            index (int): _description_
        """
        item = self.lang_data[index]
        item_id = item['question_id']
        # item_id = ''.join([i for i in item_id if i.isdigit()])
        # item_id = int(item_id[:-1].lstrip('0') + item_id[-1])
        scan_id =  item['scene_id']
        if not self.is_test:
            tgt_object_id_list = item['object_ids']
            tgt_object_name_list = item['object_names']
            answer_list = item['answers']
            answer_id_list = [self.answer_vocab.stoi(answer) 
                              for answer in answer_list if self.answer_vocab.stoi(answer) >= 0]
        else:
            tgt_object_id_list = []
            tgt_object_name_list = []
            answer_list = []
            answer_id_list = []
        question = item['question']

         # load pcds and labels
        if self.pc_type == 'gt':
            obj_pcds = self.scan_data[scan_id]['obj_pcds'] # N, 6
            obj_labels = self.scan_data[scan_id]['inst_labels'] # N
        elif self.pc_type == 'pred':
            obj_pcds = self.scan_data[scan_id]['obj_pcds_pred']
            obj_labels = self.scan_data[scan_id]['inst_labels_pred']
            # get obj labels by matching
            if not self.is_test:
                gt_obj_labels = self.scan_data[scan_id]['inst_labels'] # N
                obj_center = self.scan_data[scan_id]['obj_center']
                obj_box_size = self.scan_data[scan_id]['obj_box_size']
                obj_center_pred = self.scan_data[scan_id]['obj_center_pred']
                obj_box_size_pred = self.scan_data[scan_id]['obj_box_size_pred']
                for i, _ in enumerate(obj_center_pred):
                    for j, _ in enumerate(obj_center):
                        if eval_ref_one_sample(construct_bbox_corners(obj_center[j],
                                                                      obj_box_size[j]),
                                            construct_bbox_corners(obj_center_pred[i],
                                                                   obj_box_size_pred[i])) >= 0.25:
                            obj_labels[i] = gt_obj_labels[j]
                            break

        # filter out background or language
        if self.filter_lang:
            if self.pc_type == 'gt':
                selected_obj_idxs = [i for i, obj_label in enumerate(obj_labels)
                                    if (self.int2cat[obj_label] not in ['wall', 'floor', 'ceiling'])
                                    and (self.int2cat[obj_label] in question)]
                for _id in tgt_object_id_list:
                    if _id not in selected_obj_idxs:
                        selected_obj_idxs.append(_id)
            else:
                selected_obj_idxs = [i for i in range(len(obj_pcds))]
        else:
            if self.pc_type == 'gt':
                selected_obj_idxs = [i for i, obj_label in enumerate(obj_labels)
                                if (self.int2cat[obj_label] not in ['wall', 'floor', 'ceiling'])]
            else:
                selected_obj_idxs = [i for i in range(len(obj_pcds))]

        obj_pcds = [obj_pcds[id] for id in selected_obj_idxs]
        obj_labels = [obj_labels[id] for id in selected_obj_idxs]

        # build tgt object id and box
        if self.pc_type == 'gt':
            tgt_object_id_list = [selected_obj_idxs.index(x) for x in tgt_object_id_list]
            tgt_object_label_list = [obj_labels[x] for x in tgt_object_id_list]
            for i, _ in enumerate(tgt_object_label_list):
                assert self.int2cat[tgt_object_label_list[i]] == tgt_object_name_list[i]
        elif self.pc_type == 'pred':
            # build gt box
            gt_center = []
            gt_box_size = []
            for cur_id in tgt_object_id_list:
                gt_pcd = self.scan_data[scan_id]["obj_pcds"][cur_id]
                center, box_size = convert_pc_to_box(gt_pcd)
                gt_center.append(center)
                gt_box_size.append(box_size)

            # start filtering
            tgt_object_id_list = []
            tgt_object_label_list = []
            for i, _ in enumerate(obj_pcds):
                obj_center, obj_box_size = convert_pc_to_box(obj_pcds[i])
                for j, _ in enumerate(gt_center):
                    if eval_ref_one_sample(construct_bbox_corners(obj_center,
                                                                  obj_box_size),
                                           construct_bbox_corners(gt_center[j],
                                                                  gt_box_size[j])) >= 0.25:
                        tgt_object_id_list.append(i)
                        tgt_object_label_list.append(self.cat2int[tgt_object_name_list[j]])
                        break
        assert(len(obj_pcds) == len(obj_labels))

        # crop objects
        if self.max_obj_len < len(obj_labels):
            selected_obj_idxs = tgt_object_id_list.copy()
            remained_obj_idx = []
            for kobj, klabel in enumerate(obj_labels):
                if kobj not in  tgt_object_id_list:
                    if klabel in tgt_object_label_list:
                        selected_obj_idxs.append(kobj)
                    else:
                        remained_obj_idx.append(kobj)
                if len(selected_obj_idxs) == self.max_obj_len:
                    break
            if len(selected_obj_idxs) < self.max_obj_len:
                random.shuffle(remained_obj_idx)
                selected_obj_idxs += remained_obj_idx[:(self.max_obj_len - len(selected_obj_idxs))]
            obj_pcds = [obj_pcds[i] for i in selected_obj_idxs]
            obj_labels = [obj_labels[i] for i in selected_obj_idxs]
            tgt_object_id_list = [i for i in range(len(tgt_object_id_list))]
            assert len(obj_pcds) == self.max_obj_len

        # rebuild tgt_object_id
        if len(tgt_object_id_list) == 0:
            tgt_object_id_list.append(len(obj_pcds))
            tgt_object_label_list.append(5)

        obj_fts, obj_locs, obj_boxes, obj_labels = self._obj_processing_post(obj_pcds, obj_labels,
                                                                             is_need_bbox=True)

        # convert answer format
        answer_label = torch.zeros(self.num_answers)
        for _id in answer_id_list:
            answer_label[_id] = 1
        # tgt object id
        tgt_object_id = torch.zeros(len(obj_fts) + 1) # add 1 for pad place holder
        for _id in tgt_object_id_list:
            tgt_object_id[_id] = 1
        # tgt object sematic
        if self.sem_type == '607':
            tgt_object_label = torch.zeros(607)
        else:
            raise NotImplementedError("semantic type " + self.sem_type) 
        for _id in tgt_object_label_list:
            tgt_object_label[_id] = 1

        data_dict = {
            "sentence": question,
            "scan_dir": os.path.join(self.base_dir, 'scans'),
            "scan_id": scan_id,
            "answers": "[answer_seq]".join(answer_list),
            "answer_label": answer_label.float(), # A
            "tgt_object_id": tgt_object_id.float(), # N
            "tgt_object_label": tgt_object_label.float(), # L
            "obj_fts": obj_fts,
            "obj_locs": obj_locs,
            "obj_labels": obj_labels,
            "obj_boxes": obj_boxes, # N, 6 
            "data_idx": item_id
        }

        return data_dict

    def _load_lang(self):
        lang_data = []
        scan_ids = set()
        scan_to_item_idxs = collections.defaultdict(list)

        anno_file = os.path.join(self.base_dir,
                                 f'annotations/qa/ScanQA_v1.0_{self.split}.json')

        json_data = json.load(open(anno_file, 'r', encoding='utf-8'))
        for item in json_data:
            if self.use_unanswer or (len(set(item['answers']) & set(self.answer_cands)) > 0):
                scan_ids.add(item['scene_id'])
                scan_to_item_idxs[item['scene_id']].append(len(lang_data))
                lang_data.append(item)
        print(f'{self.split} unanswerable question {len(json_data) - len(lang_data)},'
              + f'answerable question {len(lang_data)}')
        return lang_data, scan_ids, scan_to_item_idxs

    def build_answer(self):
        train_data = json.load(open(os.path.join(self.base_dir,
                                'annotations/qa/ScanQA_v1.0_train.json'), encoding='utf-8'))
        answer_counter = sum([data['answers'] for data in train_data], [])
        answer_counter = collections.Counter(sorted(answer_counter))
        num_answers = len(answer_counter)
        answer_cands = answer_counter.keys()
        answer_vocab = ScanQAAnswer(answer_cands)
        print(f"total answers is {num_answers}")
        return num_answers, answer_vocab, answer_cands
