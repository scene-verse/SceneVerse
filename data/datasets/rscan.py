import os
import collections
import json
import random

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset

from ..build import DATASET_REGISTRY
from ..data_utils import LabelConverter, build_rotate_mat
from .base import ScanBase


@DATASET_REGISTRY.register()
class RScanPretrainObj(ScanBase):
    def __init__(self, cfg, split):
        super(RScanPretrainObj, self).__init__(cfg, split)
        self.base_dir = cfg.data.rscan_base

        self.load_scene_pcds = cfg.data.args.get('load_scene_pcds', False)
        if self.load_scene_pcds:
            self.max_pcd_num_points = cfg.data.args.get('max_pcd_num_points', None)
            assert self.max_pcd_num_points is not None
        self.bg_points_num = cfg.data.args.get('bg_points_num', 1000)

        self.scan_ids = sorted(list(self._load_split(self.split)))
        if cfg.debug.flag and cfg.debug.debug_size != -1:
            self.scan_ids = self.scan_ids[:cfg.debug.debug_size]

        print(f"Loading 3RScan {split}-set scans")
        self.scan_data = self._load_scan(self.scan_ids, is_rscan=True)
        self.scan_ids = sorted(list(self.scan_data.keys()))
        print(f"Finish loading 3RScan {split}-set scans of length {len(self.scan_ids)}")

    def __len__(self):
        return len(self.scan_ids)

    def __getitem__(self, index):
        """Data dict post-processing, for example, filtering, crop, nomalization,
        rotation, etc.

        Args:
            index (int): _description_
        """
        data_dict = self._getitem_obj_pretrain(index)
        dataset = 'rscan'
        data_dict['source'] = dataset
        return data_dict


@DATASET_REGISTRY.register()
class RScanSpatialRefer(ScanBase):
    def __init__(self, cfg, split):
        super(RScanSpatialRefer, self).__init__(cfg, split)
        self.base_dir = cfg.data.rscan_base
        self.max_obj_len = cfg.data.args.max_obj_len - 1
        self.filter_lang = cfg.data.args.filter_lang

        self.load_scene_pcds = cfg.data.args.get('load_scene_pcds', False)
        if self.load_scene_pcds:
            self.max_pcd_num_points = cfg.data.args.get('max_pcd_num_points', None)
            assert self.max_pcd_num_points is not None
        self.bg_points_num = cfg.data.args.get('bg_points_num', 1000)

        sources = cfg.data.get(self.__class__.__name__).get(split).sources
        all_scan_ids = self._load_split(self.split)

        print(f"Loading 3RScanSpatialRefer {split}-set language")
        self.lang_data, self.scan_ids = self._load_lang(cfg, all_scan_ids, sources)
        print(f"Finish loading 3RScanSpatialRefer {split}-set language of size {self.__len__()}")

        print(f"Loading 3RScan {split}-set scans")
        self.scan_data = self._load_scan(self.scan_ids, is_rscan=True)
        print(f"Finish loading 3RScan {split}-set scans")

         # build unique multiple look up
        for scan_id in self.scan_ids:
            inst_labels = self.scan_data[scan_id]['inst_labels']
            self.scan_data[scan_id]['label_count'] = collections.Counter(
                                    [l for l in inst_labels])

    def __len__(self):
        return len(self.lang_data)

    def __getitem__(self, index):
        """Data dict post-processing, for example, filtering, crop, nomalization,
        rotation, etc.

        Args:
            index (int): _description_
        """
        data_dict = self._getitem_refer(index)
        return data_dict


class RScanBase(ScanBase):
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.split = split
        assert self.split in ['train', 'val', 'test']
        self.base_dir = cfg.data.rscan_base
        self.scannet_dir = cfg.data.scan_family_base

        self.int2cat = json.load(open(os.path.join(self.scannet_dir,
                                            "annotations/meta_data/scannetv2_raw_categories.json"),
                                            'r', encoding="utf-8"))
        self.cat2int = {w: i for i, w in enumerate(self.int2cat)}
        self.label_converter = LabelConverter(os.path.join(self.scannet_dir,
                                            "annotations/meta_data/scannetv2-labels.combined.tsv"))

    def _load_split(self, split):
        split_file = os.path.join(self.base_dir, '3RScan-anno/splits/' + split + "_split.txt")
        split_scan_ids = {x.strip() for x in open(split_file, 'r', encoding="utf-8")}
        split_scan_ids = sorted(split_scan_ids)

        return split_scan_ids

    def _load_rscan(self, scan_ids):
        scans = {}
        for scan_id in tqdm(scan_ids):
            if not os.path.exists(os.path.join(self.base_dir, "3RScan-ours-align",
                                               scan_id, "pcd-align.pth")):
                continue
            pcd_data = torch.load(os.path.join(self.base_dir, "3RScan-ours-align",
                                               scan_id, "pcd-align.pth"))
            points, colors, instance_labels = pcd_data[0], pcd_data[1], pcd_data[2]
            colors = colors / 127.5 - 1
            pcds = np.concatenate([points, colors], 1)
            # build obj_pcds
            inst_to_label = torch.load(os.path.join(self.base_dir, "3RScan-ours-align", scan_id,
                                                    "inst_to_label.pth"))
            obj_pcds = []
            inst_ids = []
            inst_labels = []
            obj_val = []
            bg_indices = np.full((points.shape[0], ), 1, dtype=np.bool_)
            for inst_id in inst_to_label.keys():
                if inst_to_label[inst_id] in self.cat2int.keys():
                    mask = instance_labels == inst_id
                    if np.sum(mask) == 0:
                        continue
                    obj_pcds.append(pcds[mask])
                    inst_ids.append(inst_id)
                    inst_labels.append(self.cat2int[inst_to_label[inst_id]])
                    obj_val.append(inst_to_label[inst_id] in ['wall', 'floor', 'ceiling'])
                    if inst_to_label[inst_id] not in ['wall', 'floor', 'ceiling']:
                        bg_indices[mask] = False
            if all(obj_val):
                continue
            scans[scan_id] = {}
            # scans[scan_id]['scene_pcds'] = pcds
            scans[scan_id]['obj_pcds'] = obj_pcds
            scans[scan_id]['inst_labels'] = inst_labels
            scans[scan_id]['inst_ids'] = inst_ids
            scans[scan_id]['bg_pcds'] = pcds[bg_indices]
        return scans

    def _load_lang(self, cfg, scan_ids, caption_source):
        json_data = []
        lang_data = []
        valid_scan_ids = []
        if caption_source:
            for anno_type in caption_source:
                if 'template' == anno_type:
                    anno_file = os.path.join(self.base_dir, '3RScan-anno/template_gen_language.json')
                    json_data.extend(json.load(open(anno_file, 'r', encoding='utf-8')))
                elif 'gpt' == anno_type:
                    anno_file = os.path.join(self.base_dir, '3RScan-anno/gpt_gen_language.json')
                    json_data.extend(json.load(open(anno_file, 'r', encoding='utf-8')))
                else:
                    anno_file = os.path.join(self.base_dir, f'3RScan-anno/ssg_ref_{anno_type}.json')
                    json_data.extend(json.load(open(anno_file, 'r', encoding='utf-8')))
        for item in json_data:
            if item['scan_id'] in scan_ids:
                lang_data.append(item)
                if item['scan_id'] not in valid_scan_ids:
                    valid_scan_ids.append(item['scan_id'])
        valid_scan_ids = sorted(valid_scan_ids)
        if cfg.debug.flag and cfg.debug.debug_size != -1:
            valid_scan_ids = valid_scan_ids[:cfg.debug.debug_size]
            lang_data = [item for item in lang_data if item['scan_id'] in valid_scan_ids]

        return lang_data, valid_scan_ids

    def _obj_processing_post(self, obj_pcds, obj_labels, is_need_bbox=False, rot_aug=False):
        # rotate obj
        rot_matrix = build_rotate_mat(self.split, rot_aug)

        # normalize pc and calculate location
        obj_fts = []
        obj_locs = []
        obj_boxes = []
        for obj_pcd in obj_pcds:
            # build locs
            if rot_matrix is not None:
                obj_pcd[:, :3] = np.matmul(obj_pcd[:, :3], rot_matrix.transpose())
            obj_center = obj_pcd[:, :3].mean(0)
            obj_size = obj_pcd[:, :3].max(0) - obj_pcd[:, :3].min(0)
            obj_locs.append(np.concatenate([obj_center, obj_size], 0))

            # build box
            if is_need_bbox:
                obj_box_center = (obj_pcd[:, :3].max(0) + obj_pcd[:, :3].min(0)) / 2
                obj_box_size = obj_pcd[:, :3].max(0) - obj_pcd[:, :3].min(0)
                obj_boxes.append(np.concatenate([obj_box_center, obj_box_size], 0))

            # subsample
            pcd_idxs = np.random.choice(len(obj_pcd), size=self.num_points,
                                        replace=len(obj_pcd) < self.num_points)
            obj_pcd = obj_pcd[pcd_idxs]
            # normalize
            obj_pcd[:, :3] = obj_pcd[:, :3] - obj_pcd[:, :3].mean(0)
            max_dist = np.max(np.sqrt(np.sum(obj_pcd[:, :3]**2, 1)))
            if max_dist < 1e-6: # take care of tiny point-clouds, i.e., padding
                max_dist = 1
            obj_pcd[:, :3] = obj_pcd[:, :3] / max_dist
            obj_fts.append(obj_pcd)

        # convert to torch
        obj_fts = torch.from_numpy(np.stack(obj_fts, 0))
        obj_locs = torch.from_numpy(np.array(obj_locs))
        obj_boxes = torch.from_numpy(np.array(obj_boxes))
        obj_labels = torch.LongTensor(obj_labels)

        assert obj_labels.shape[0] == obj_locs.shape[0]
        assert obj_fts.shape[0] == obj_locs.shape[0]

        return obj_fts, obj_locs, obj_boxes, obj_labels

    def _obj_processing_aug(self, obj_pcds, obj_labels, is_need_bbox=False):
        # calculate size
        if self.augmentor:
            data_dict = self.augmentor.forward({'obj_pcds': obj_pcds,
                                                'num_points': self.num_points}
                                               )
        obj_pcds = data_dict['obj_pcds']
        obj_sizes = data_dict['obj_sizes']
        # normalize pc and calculate location
        obj_fts = []
        obj_locs = []
        obj_boxes = []
        for _i, obj_pcd in enumerate(obj_pcds):
            # build locs
            obj_center = obj_pcd[:, :3].mean(0)
            obj_size = obj_sizes[_i]
            obj_locs.append(np.concatenate([obj_center, obj_size], 0))

            # build box
            if is_need_bbox:
                obj_box_center = (obj_pcd[:, :3].max(0) + obj_pcd[:, :3].min(0)) / 2
                obj_box_size = obj_sizes[_i]
                obj_boxes.append(np.concatenate([obj_box_center, obj_box_size], 0))

            # normalize
            obj_pcd[:, :3] = obj_pcd[:, :3] - obj_pcd[:, :3].mean(0)
            max_dist = np.max(np.sqrt(np.sum(obj_pcd[:, :3]**2, 1)))
            if max_dist < 1e-6: # take care of tiny point-clouds, i.e., padding
                max_dist = 1
            obj_pcd[:, :3] = obj_pcd[:, :3] / max_dist
            obj_fts.append(obj_pcd)

        # convert to torch
        obj_fts = torch.from_numpy(np.stack(obj_fts, 0))
        obj_locs = torch.from_numpy(np.array(obj_locs))
        obj_boxes = torch.from_numpy(np.array(obj_boxes))
        obj_labels = torch.LongTensor(obj_labels)

        assert obj_labels.shape[0] == obj_locs.shape[0]
        assert obj_fts.shape[0] == obj_locs.shape[0]

        return obj_fts, obj_locs, obj_boxes, obj_labels
    
    def _scene_processing_aug(self, obj_pcds, bg_pcds, obj_labels, is_need_bbox=False):
        # sample background points
        fg_points_num = len(obj_pcds) * self.num_points
        assert fg_points_num < self.max_pcd_num_points
        bg_points_num = min(self.max_pcd_num_points - fg_points_num, self.bg_points_num)
        assert len(bg_pcds) > 0
        assert bg_points_num > 0
        bg_points_indices = np.random.choice(len(bg_pcds), size=bg_points_num, replace=len(bg_pcds) < bg_points_num)
        bg_pcds = bg_pcds[bg_points_indices]

        # augment objects
        if self.augmentor:
            data_dict = self.augmentor.forward({'obj_pcds': obj_pcds, 
                                                'bg_pcds': bg_pcds, 
                                                'num_points': self.num_points})

        obj_pcds = data_dict['obj_pcds']
        obj_sizes = data_dict['obj_sizes']
        bg_pcds = data_dict['bg_pcds']
        assert len(obj_pcds) * obj_pcds[0].shape[0] == fg_points_num

        # calculate location and generate scene_pcd
        obj_locs = []
        obj_boxes = []
        scene_pcds = []
        for _i, obj_pcd in enumerate(obj_pcds):
            # build locs
            obj_center = obj_pcd[:, :3].mean(0)
            obj_size = obj_sizes[_i]
            obj_locs.append(np.concatenate([obj_center, obj_size], 0))

            # build box
            if is_need_bbox:
                obj_box_center = (obj_pcd[:, :3].max(0) + obj_pcd[:, :3].min(0)) / 2
                obj_box_size = obj_sizes[_i]
                obj_boxes.append(np.concatenate([obj_box_center, obj_box_size], 0))

            # build scene pcd
            scene_pcds.extend(obj_pcd)

        # # sample background points
        # assert len(scene_pcds) < self.max_pcd_num_points
        # bg_points_num = min(self.max_pcd_num_points - len(scene_pcds), self.bg_points_num)
        # assert len(bg_pcds) > 0
        # assert bg_points_num > 0
        # bg_points_indices = np.random.choice(len(bg_pcds), size=bg_points_num, replace=len(bg_pcds) < bg_points_num)
        # scene_pcds.extend(bg_pcds[bg_points_indices])
        # assert len(scene_pcds) == self.max_pcd_num_points

        scene_pcds.extend(bg_pcds)

        # generate obj point indices masks
        obj_pcds_masks = []
        offset = 0
        for _j in range(len(obj_pcds)):
            mask = np.arange(self.num_points) + offset
            assert len(mask) == len(obj_pcds[_j])
            obj_pcds_masks.append(mask)
            offset += self.num_points

        # convert to torch
        obj_locs = torch.from_numpy(np.array(obj_locs))
        obj_boxes = torch.from_numpy(np.array(obj_boxes))
        obj_labels = torch.LongTensor(obj_labels)
        obj_pcds_masks = torch.from_numpy(np.array(obj_pcds_masks))
        scene_pcds = np.array(scene_pcds)

        assert obj_labels.shape[0] == obj_locs.shape[0]
        assert obj_pcds_masks.shape[0] == obj_locs.shape[0]

        return obj_locs, obj_boxes, obj_labels, obj_pcds_masks, scene_pcds
