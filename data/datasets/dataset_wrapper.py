import os
from time import time

import torch
import numpy as np
from fvcore.common.registry import Registry
from transformers import BertTokenizer, T5Tokenizer, AutoTokenizer
from torch.utils.data import Dataset, default_collate
import random
import MinkowskiEngine as ME
import copy

from ..data_utils import random_word, random_point_cloud, pad_tensors, Vocabulary, random_caption_word
# from modules.third_party.softgroup_ops.ops import functions as sg_ops


DATASETWRAPPER_REGISTRY = Registry("dataset_wrapper")
DATASETWRAPPER_REGISTRY.__doc__ = """ """


@DATASETWRAPPER_REGISTRY.register()
class MaskDatasetWrapper(Dataset):
    def __init__(self, cfg, dataset, split="train"):
        # tokenizer, max_seq_length=80, max_obj_len=80,
        #  mask_strategy='random', txt_mask_ratio=0.15, pc_mask_ratio=0.1
        assert cfg.data.args.mask_strategy in ['random']
        self.dataset = dataset
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        self.max_seq_length = cfg.data.args.max_seq_len
        self.max_obj_len = cfg.data.args.max_obj_len
        self.txt_mask_ratio = cfg.data.args.txt_mask_ratio
        self.pc_mask_ratio = cfg.data.args.pc_mask_ratio

        self.use_voxel = cfg.data.args.get('use_voxel', None)
        if self.use_voxel:
            self.voxel_size = cfg.data.args.get('voxel_size', 0.02)

        self.use_scene_cap = cfg.data.args.get("use_scene_cap", False)
        self.max_scene_cap_len = cfg.data.args.get("max_scene_cap_len", self.max_seq_length)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data_dict = self.dataset[idx]
        sentence = data_dict['sentence']
        encoded_input = self.tokenizer(sentence, max_length=self.max_seq_length,
                          add_special_tokens=True, truncation=True,
                          padding='max_length', return_tensors="pt")
        # build txt
        data_dict['txt_ids'] = encoded_input['input_ids'].squeeze(0) # L
        data_dict['txt_masks'] = encoded_input['attention_mask'].squeeze(0) # L

        if self.use_scene_cap:
            scene_cap = data_dict["scene_cap"] + " " + sentence
            encoded_scene_cap = self.tokenizer(scene_cap, max_length=self.max_scene_cap_len,
                          add_special_tokens=True, truncation=True,
                          padding='max_length', return_tensors="pt")
            data_dict['scene_txt_ids'] = encoded_scene_cap['input_ids'].squeeze(0)          # L
            data_dict['scene_txt_masks'] = encoded_scene_cap['attention_mask'].squeeze(0)   # L

        # mask txt
        masked_txt_ids, masked_lm_labels = random_word(data_dict['txt_ids'], data_dict['txt_masks'],
                                                       self.tokenizer, self.txt_mask_ratio)
        data_dict['txt_ids'] = masked_txt_ids
        data_dict['masked_lm_labels'] = masked_lm_labels
        # build object
        data_dict['obj_masks'] = (torch.arange(self.max_obj_len) < len(data_dict['obj_locs'])) # O
        if 'obj_fts' in data_dict.keys():
            data_dict['obj_fts'] = pad_tensors(data_dict['obj_fts'], lens=self.max_obj_len,
                                                    pad=1.0).float() # O, 1024, 6
        if 'obj_pcds_masks' in data_dict.keys():
            data_dict['obj_pcds_masks'] = pad_tensors(data_dict['obj_pcds_masks'], lens=self.max_obj_len, 
                                                      pad=1.0).float()
        data_dict['obj_locs']= pad_tensors(data_dict['obj_locs'], lens=self.max_obj_len,
                                                pad=0.0).float() # O, 3
        data_dict['obj_labels'] = pad_tensors(data_dict['obj_labels'], lens=self.max_obj_len,
                                                   pad=-100).long() # O
        # mask object, 0 means mask object, 1 means keep object
        if 'obj_fts' in data_dict.keys():
            obj_sem_masks = random_point_cloud(data_dict['obj_fts'], data_dict['obj_masks'],
                                            self.pc_mask_ratio)
            data_dict['obj_sem_masks'] = obj_sem_masks
        else:
            obj_sem_masks = []
            for i in range(self.max_obj_len):
                if i >= len(data_dict['obj_locs']):
                    obj_sem_masks.append(0)
                else:
                    prob = random.random()
                    if prob < self.pc_mask_ratio:
                        obj_sem_masks.append(0)
                    else:
                        obj_sem_masks.append(1)
            data_dict['obj_sem_masks'] = torch.tensor(obj_sem_masks).long()
        if 'tgt_object_id' in data_dict.keys():
            data_dict['tgt_object_id'] = data_dict['tgt_object_id'].long() # 1 or O

        # # Scene pcds
        # data_dict["scene_pcds"] = torch.from_numpy(data_dict["scene_pcds"]).float()
        key_list = [
            'txt_ids', 'txt_masks', 'masked_lm_labels', 'obj_masks', 'obj_fts',
            'obj_locs', 'obj_labels', 'obj_sem_masks', 'tgt_object_id'
        ]
        if 'obj_fts' not in data_dict.keys():
            key_list.remove('obj_fts')
            # key_list.remove('obj_sem_masks')
        if 'obj_pcds_masks' in data_dict.keys():
            key_list.append('obj_pcds_masks')
        if 'scene_pcds' in data_dict.keys():
            key_list.append('scene_pcds')
        if 'scene_txt_ids' in data_dict.keys():
            key_list.append('scene_txt_ids')
        if 'scene_txt_masks' in data_dict.keys():
            key_list.append('scene_txt_masks')
        data_dict = {k : v for k, v in data_dict.items() if k in key_list}
        return data_dict
    
    def collate_fn(self, batch_list):
        ret = default_collate(batch_list)
        return ret


@DATASETWRAPPER_REGISTRY.register()
class ScanFamilyDatasetWrapperOld(Dataset):
    def __init__(self, cfg, dataset, split="train"):
        # stokenizer, max_seq_length=80, max_obj_len=80
        self.dataset = dataset
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        self.max_seq_length = cfg.data.args.max_seq_len
        self.max_obj_len = cfg.data.args.max_obj_len

        self.use_voxel = cfg.data.args.get('use_voxel', None)
        if self.use_voxel:
            self.voxel_size = cfg.data.args.get('voxel_size', 0.02)

        self.use_scene_cap = cfg.data.args.get("use_scene_cap", False)
        self.max_scene_cap_len = cfg.data.args.get("max_scene_cap_len", self.max_seq_length)

    def __len__(self):
        return len(self.dataset)

    def pad_tensors(self, tensors, lens=None, pad=0):
        assert tensors.shape[0] <= lens
        if tensors.shape[0] == lens:
            return tensors
        shape = list(tensors.shape)
        shape[0] = lens - shape[0]
        res = torch.ones(shape, dtype=tensors.dtype) * pad
        res = torch.cat((tensors, res), dim=0)
        return res

    def __getitem__(self, idx):
        data_dict = self.dataset[idx]
        sentence = data_dict['sentence']
        encoded_input = self.tokenizer(sentence, max_length=self.max_seq_length,
                          add_special_tokens=True, truncation=True,
                          padding='max_length', return_tensors="pt")
        # build txt
        data_dict['txt_ids'] = encoded_input['input_ids'].squeeze(0) # L
        data_dict['txt_masks'] = encoded_input['attention_mask'].squeeze(0) # L
        if self.use_scene_cap:
            scene_cap = data_dict["scene_cap"] + " " + sentence
            encoded_scene_cap = self.tokenizer(scene_cap, max_length=self.max_scene_cap_len,
                          add_special_tokens=True, truncation=True,
                          padding='max_length', return_tensors="pt")
            data_dict['scene_txt_ids'] = encoded_scene_cap['input_ids'].squeeze(0)          # L
            data_dict['scene_txt_masks'] = encoded_scene_cap['attention_mask'].squeeze(0)   # L
        # build object
        data_dict['obj_masks'] = (torch.arange(self.max_obj_len) < len(data_dict['obj_locs'])) # O
        if 'obj_fts' in data_dict.keys():
            data_dict['obj_fts'] = self.pad_tensors(data_dict['obj_fts'], lens=self.max_obj_len,
                                                    pad=1.0).float() # O, 1024, 6
        if 'obj_pcds_masks' in data_dict.keys():
            data_dict['obj_pcds_masks'] = pad_tensors(data_dict['obj_pcds_masks'], lens=self.max_obj_len, 
                                                      pad=1.0).float()
        data_dict['obj_locs']= self.pad_tensors(data_dict['obj_locs'], lens=self.max_obj_len,
                                                pad=0.0).float() # O, 3
        data_dict['obj_boxes']= self.pad_tensors(data_dict['obj_boxes'], lens=self.max_obj_len,
                                                 pad=0.0).float() # O, 3
        data_dict['obj_labels'] = self.pad_tensors(data_dict['obj_labels'], lens=self.max_obj_len,
                                                   pad=-100).long() # O
        # build sem mask, no mask
        data_dict['obj_sem_masks'] = (torch.arange(self.max_obj_len) < len(data_dict['obj_locs']))
        # build label for refer
        data_dict['tgt_object_label'] = data_dict['tgt_object_label'].long() # 1 or C
        data_dict['tgt_object_id'] = data_dict['tgt_object_id'].long() # 1 or O
        if len(data_dict['tgt_object_id']) > 1: # O, pad to max objet length
            data_dict['tgt_object_id'] = self.pad_tensors(data_dict['tgt_object_id'].long(),
                                                          lens=self.max_obj_len, pad=0).long() # O
        # build target
        if data_dict.get('tgt_object_id_iou25') is not None:
            data_dict['tgt_object_id_iou25'] = self.pad_tensors(data_dict['tgt_object_id_iou25'],
                                                                lens=self.max_obj_len, pad=0).long()
        if data_dict.get('tgt_object_id_iou50') is not None:
            data_dict['tgt_object_id_iou50'] = self.pad_tensors(data_dict['tgt_object_id_iou50'],
                                                                lens=self.max_obj_len, pad=0).long()
        # build label for qa
        if "answer_label" in data_dict:
            data_dict['answer_label'] = data_dict['answer_label'].long() # N, C
        return data_dict
    
    def collate_fn(self, batch_list):
        ret = default_collate(batch_list)
        return ret

