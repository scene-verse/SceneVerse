import os
import collections
import json
import pickle
import random

import jsonlines
from tqdm import tqdm
from scipy import sparse
import numpy as np
import torch
from torch.utils.data import Dataset

from common.misc import rgetattr
from ..data_utils import convert_pc_to_box, LabelConverter, build_rotate_mat, load_matrix_from_txt, \
                        construct_bbox_corners, eval_ref_one_sample  
import einops 


SCAN_DATA = {}

class ScanNetBase(Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.split = split
        self.base_dir = cfg.data.scan_family_base
        assert self.split in ['train', 'val', 'test']

        self.int2cat = json.load(open(os.path.join(self.base_dir,
                                            "annotations/meta_data/scannetv2_raw_categories.json"),
                                            'r', encoding="utf-8"))
        self.cat2int = {w: i for i, w in enumerate(self.int2cat)}
        self.label_converter = LabelConverter(os.path.join(self.base_dir,
                                            "annotations/meta_data/scannetv2-labels.combined.tsv"))

        # self.referit3d_camera_pose = json.load(open(os.path.join(self.base_dir,
        #                                     "annotations/meta_data/scans_axis_alignment_matrices.json"),
        #                                     'r', encoding="utf-8"))
        self.rot_matrix = build_rotate_mat(self.split)
        self.use_cache = rgetattr(self.cfg.data, 'mvdatasettings.use_cache', False)
        self.cache = {}

    def __len__(self):
        return len(self.lang_data)

    def __getitem__(self, index):
        raise NotImplementedError

    def _load_one_scan(self, scan_id, pc_type = 'gt', load_inst_info = False, 
                      load_multiview_info = False, load_mask3d_voxel = False, load_pc_info = True, load_segment_info=False,
                      load_offline_segment_voxel=False, load_offline_segment_image=False, load_offline_segment_point=False, load_nocls=False):
        one_scan = {}
        if load_inst_info:
            inst_labels, inst_locs, inst_colors = self._load_inst_info(scan_id)
            one_scan['inst_labels'] = inst_labels # (n_obj, )
            one_scan['inst_locs'] = inst_locs # (n_obj, 6) center xyz, whl
            one_scan['inst_colors'] = inst_colors # (n_obj, 3x4) cluster * (weight, mean rgb)

        if load_pc_info:
            # load pcd data
            pcd_data = torch.load(os.path.join(self.base_dir, "scan_data",
                                            "pcd_with_global_alignment", f'{scan_id}.pth'))
            points, colors, instance_labels = pcd_data[0], pcd_data[1], pcd_data[-1]
            colors = colors / 127.5 - 1
            pcds = np.concatenate([points, colors], 1)
            # convert to gt object
            if load_inst_info:
                obj_pcds = []
                bg_indices = np.full((points.shape[0], ), 1, dtype=np.bool_)
                for i in range(instance_labels.max() + 1):
                    mask = instance_labels == i     # time consuming
                    obj_pcds.append(pcds[mask])
                    if not self.int2cat[inst_labels[i]] in ['wall', 'floor', 'ceiling']:
                        bg_indices[mask] = False
                one_scan['obj_pcds'] = obj_pcds
                # assert sum([len(obj_pcd) for obj_pcd in obj_pcds]) + bg_indices.sum() == points.shape[0]
                one_scan['bg_pcds'] = pcds[bg_indices]
                # calculate box for matching
                obj_center = []
                obj_box_size = []
                for obj_pcd in obj_pcds:
                    _c, _b = convert_pc_to_box(obj_pcd)
                    obj_center.append(_c)
                    obj_box_size.append(_b)
                one_scan['obj_center'] = obj_center
                one_scan['obj_box_size'] = obj_box_size

            obj_mask_path = os.path.join(self.base_dir, "mask", str(scan_id) + ".mask" + ".npz")
            if os.path.exists(obj_mask_path):
                obj_label_path = os.path.join(self.base_dir, "mask", str(scan_id) + ".label" + ".npy")
                obj_pcds = []
                obj_mask = np.array(sparse.load_npz(obj_mask_path).todense())[:50, :]
                obj_labels = np.load(obj_label_path)[:50]
                obj_l = []
                bg_indices = np.full((pcds.shape[0], ), 1, dtype=np.bool_)
                for i in range(obj_mask.shape[0]):
                    mask = obj_mask[i]
                    if pcds[mask == 1, :].shape[0] > 0:
                        obj_pcds.append(pcds[mask == 1, :])
                        obj_l.append(obj_labels[i])
                        # if not self.int2cat[obj_labels[i]] in ['wall', 'floor', 'ceiling']:
                        bg_indices[mask == 1] = False
                one_scan['obj_pcds_pred'] = obj_pcds
                one_scan['inst_labels_pred'] = obj_l
                one_scan['bg_pcds_pred'] = pcds[bg_indices]
                # calculate box for pred
                obj_center_pred = []
                obj_box_size_pred = []
                for obj_pcd in obj_pcds:
                    _c, _b = convert_pc_to_box(obj_pcd)
                    obj_center_pred.append(_c)
                    obj_box_size_pred.append(_b)
                one_scan['obj_center_pred'] = obj_center_pred
                one_scan['obj_box_size_pred'] = obj_box_size_pred

        if load_multiview_info:
            one_scan['multiview_info'] = self._load_multiview_info(scan_id)

        if load_mask3d_voxel:
            one_scan['mask3d_voxel'] = self._load_mask3d_voxel(scan_id)
        
        # load segment for mask3d
        if load_segment_info:
            one_scan["scene_pcds"] = np.load(os.path.join(self.base_dir, "scan_data", "pcd_mask3d", f'{scan_id[-7:]}.npy'))
        
        # load offline feature 
        if load_offline_segment_voxel:
            if load_nocls:
                one_scan['offline_segment_voxel'] = torch.load(os.path.join(self.base_dir, "scan_data", "mask3d_voxel_feature_nocls", f'{scan_id}.pth'))
            else:
                one_scan['offline_segment_voxel'] = torch.load(os.path.join(self.base_dir, "scan_data", "mask3d_voxel_feature", f'{scan_id}.pth'))
            
        if load_offline_segment_image:
            one_scan['offline_segment_image'] = torch.load(os.path.join(self.base_dir, "scan_data", "mask3d_image_feature", f'{scan_id}.pth'))

        return (scan_id, one_scan)

    def _load_scannet(self, scan_ids, pc_type = 'gt', load_inst_info = False, 
                      load_multiview_info = False, load_mask3d_voxel = False, load_pc_info = True, load_segment_info = False, 
                      load_offline_segment_voxel=False, load_offline_segment_image=False, load_offline_segment_point=False, load_nocls=False,
                      process_num = 1):
        unloaded_scan_ids = [scan_id for scan_id in scan_ids if scan_id not in SCAN_DATA]
        print(f"Loading scans: {len(unloaded_scan_ids)} / {len(scan_ids)}")
        scans = {}
        if process_num > 1:
            from joblib import Parallel, delayed
            res_all = Parallel(n_jobs=process_num)(
                delayed(self._load_one_scan)(scan_id, pc_type = pc_type,
                                        load_inst_info = load_inst_info,
                                        load_multiview_info = load_multiview_info,
                                        load_mask3d_voxel = load_mask3d_voxel,
                                        load_offline_segment_voxel=load_offline_segment_voxel, load_offline_segment_image=load_offline_segment_image,
                                        load_offline_segment_point=load_offline_segment_point, load_nocls=load_nocls,
                                        load_pc_info = load_pc_info, load_segment_info=load_segment_info) for scan_id in tqdm(unloaded_scan_ids))
            for scan_id, one_scan in tqdm(res_all):
                scans[scan_id] = one_scan

        else:
            for scan_id in tqdm(unloaded_scan_ids):
                _, one_scan = self._load_one_scan(scan_id, pc_type = pc_type,
                                                  load_inst_info = load_inst_info, 
                                                  load_multiview_info = load_multiview_info, 
                                                  load_mask3d_voxel = load_mask3d_voxel,
                                                  load_pc_info = load_pc_info, load_segment_info=load_segment_info,
                                                  load_offline_segment_voxel=load_offline_segment_voxel, load_offline_segment_image=load_offline_segment_image,
                                                  load_offline_segment_point=load_offline_segment_point, load_nocls=load_nocls)
                scans[scan_id] = one_scan

        SCAN_DATA.update(scans)
        scans = {scan_id: SCAN_DATA[scan_id] for scan_id in scan_ids}
        return scans

    def _load_lang(self, cfg):
        caption_source = cfg.sources
        lang_data = []
        if caption_source:
            if 'scanrefer' in caption_source:
                anno_file = os.path.join(self.base_dir, 'annotations/refer/scanrefer.jsonl')
                with jsonlines.open(anno_file, 'r') as _f:
                    for item in _f:
                        if item['scan_id'] in self.scannet_scan_ids:
                            lang_data.append(('scannet', item['scan_id'], item['utterance']))

            if 'referit3d' in caption_source:
                for anno_type in cfg.referit3d.anno_type:
                    anno_file = os.path.join(self.base_dir,
                                             f'annotations/refer/{anno_type}.jsonl')
                    with jsonlines.open(anno_file, 'r') as _f:
                        for item in _f:
                            if item['scan_id'] in self.scannet_scan_ids:
                                lang_data.append(('scannet', item['scan_id'], item['utterance']))

            if 'scanqa' in caption_source:
                anno_file_list = ['annotations/qa/ScanQA_v1.0_train.json',
                                  'annotations/qa/ScanQA_v1.0_val.json']
                for anno_file in anno_file_list:
                    anno_file = os.path.join(self.base_dir, anno_file)
                    json_data = json.load(open(anno_file, 'r', encoding='utf-8'))
                    for item in json_data:
                        if item['scene_id'] in self.scannet_scan_ids:
                            for i in range(len(item['answers'])):
                                lang_data.append(('scannet', item['scene_id'],
                                                  item['question'] + " " + item['answers'][i]))

            if 'sgrefer' in caption_source:
                for anno_type in cfg.sgrefer.anno_type:
                    anno_file = os.path.join(self.base_dir,
                                             f'annotations/refer/ssg_ref_{anno_type}.json')
                    json_data = json.load(open(anno_file, 'r', encoding='utf-8'))
                    for item in json_data:
                        if item['scan_id'] in self.scannet_scan_ids:
                            lang_data.append(('scannet', item['scan_id'], item['utterance']))

            if 'sgcaption' in caption_source:
                for anno_type in cfg.sgcaption.anno_type:
                    anno_file = os.path.join(self.base_dir,
                                             f'annotations/refer/ssg_caption_{anno_type}.json')
                    json_data = json.load(open(anno_file, 'r', encoding='utf-8'))
                    for item in json_data:
                        if item['scan_id'] in self.scannet_scan_ids:
                            lang_data.append(('scannet', item['scan_id'], item['utterance']))
        return lang_data

    def _load_split(self, cfg, split, use_multi_process = False):
        if use_multi_process and split in ['train']:
            split_file = os.path.join(self.base_dir, 'annotations/splits/scannetv2_'+ split + "_sort.json")
            with open(split_file, 'r') as f:
                scannet_scan_ids = json.load(f)
        else:
            split_file = os.path.join(self.base_dir, 'annotations/splits/scannetv2_'+ split + ".txt")
            scannet_scan_ids = {x.strip() for x in open(split_file, 'r', encoding="utf-8")}
        scannet_scan_ids = sorted(scannet_scan_ids)

        if cfg.debug.flag and cfg.debug.debug_size != -1:
            scannet_scan_ids = list(scannet_scan_ids)[:cfg.debug.debug_size]
        return scannet_scan_ids

    def _load_inst_info(self, scan_id):
        # inst_labels = json.load(open(os.path.join(self.base_dir, 'scan_data',
        #                                           'instance_id_to_label',
        #                                          f'{scan_id}.json'), encoding="utf-8"))
        inst_labels = torch.load(os.path.join(self.base_dir, 'scan_data',
                                                'instance_id_to_label',
                                                f'{scan_id}.pth'))
        inst_labels = [self.cat2int[i] for i in inst_labels.values()]

        inst_loc_path = os.path.join(self.base_dir, 'scan_data',
                                     'instance_id_to_loc', f'{scan_id}.npy')
        if os.path.exists(inst_loc_path):
            inst_locs = np.load(inst_loc_path)
        else:
            inst_locs = None

        inst_color_path = os.path.join(self.base_dir, 'scan_data',
                                       'instance_id_to_gmm_color', f'{scan_id}.json') 
        if os.path.exists(inst_color_path):
            inst_colors = json.load(open(inst_color_path, encoding="utf-8"))
            inst_colors = [np.concatenate(
                [np.array(x['weights'])[:, None], np.array(x['means'])],
                axis=1).astype(np.float32) for x in inst_colors]
        else:
            inst_colors = None

        return inst_labels, inst_locs, inst_colors

    def _obj_processing_post(self, obj_pcds, obj_labels, is_need_bbox=False, rot_aug=True):
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
        # augment objects
        if self.augmentor:
            data_dict = self.augmentor.forward({'obj_pcds': obj_pcds,
                                                'num_points': self.num_points})

        obj_pcds = data_dict['obj_pcds']
        if isinstance(obj_pcds, list):
            obj_pcds = torch.Tensor(np.array(obj_pcds))
        obj_sizes = torch.Tensor(np.array(data_dict['obj_sizes']))

        xyz = obj_pcds[:, :, :3]
        center = xyz.mean(1)
        xyz_min = xyz.min(1).values
        xyz_max = xyz.max(1).values
        box_center = (xyz_min + xyz_max) / 2
        size = torch.Tensor(obj_sizes)
        # size = xyz_max - xyz_min
        obj_locs = torch.cat([center, size], dim=1)
        obj_boxes = torch.cat([box_center, size], dim=1)

        # centering
        obj_pcds[:, :, :3].sub_(obj_pcds[:, :, :3].mean(1, keepdim=True))

        # normalization
        max_dist = (obj_pcds[:, :, :3]**2).sum(2).sqrt().max(1).values
        max_dist.clamp_(min=1e-6)
        obj_pcds[:, :, :3].div_(max_dist[:, None, None])

        # convert to torch
        obj_labels = torch.LongTensor(obj_labels)

        assert obj_labels.shape[0] == obj_locs.shape[0]

        return obj_pcds, obj_locs, obj_boxes, obj_labels
    
    def _scene_processing_aug(self, obj_pcds, bg_pcds, obj_labels, is_need_bbox=False):
        obj_len = len(obj_pcds)
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
                                                'bg_pcds': torch.Tensor(bg_pcds), 
                                                'num_points': self.num_points})

        obj_pcds = data_dict['obj_pcds']
        if isinstance(obj_pcds, list):
            obj_pcds = torch.Tensor(np.array(obj_pcds))
        obj_sizes = torch.Tensor(np.array(data_dict['obj_sizes']))
        bg_pcds = data_dict['bg_pcds']
        assert len(obj_pcds) * obj_pcds[0].shape[0] == fg_points_num

        scene_pcds = np.vstack([np.array(obj_pcds.reshape(-1, 6)), np.array(bg_pcds)])

        xyz = obj_pcds[:, :, :3]
        center = xyz.mean(1)
        xyz_min = xyz.min(1).values
        xyz_max = xyz.max(1).values
        box_center = (xyz_min + xyz_max) / 2
        size = torch.Tensor(obj_sizes)
        # size = xyz_max - xyz_min
        obj_locs = torch.cat([center, size], dim=1)
        obj_boxes = torch.cat([box_center, size], dim=1)

        # centering
        obj_pcds[:, :, :3].sub_(obj_pcds[:, :, :3].mean(1, keepdim=True))

        # normalization
        max_dist = (obj_pcds[:, :, :3]**2).sum(2).sqrt().max(1).values
        max_dist.clamp_(min=1e-6)
        obj_pcds[:, :, :3].div_(max_dist[:, None, None])

        # generate obj point indices masks
        obj_pcds_masks = []
        offset = 0
        for _j in range(obj_len):
            mask = np.arange(self.num_points) + offset
            assert len(mask) == len(obj_pcds[_j])
            obj_pcds_masks.append(mask)
            offset += self.num_points

        # convert to torch
        obj_labels = torch.LongTensor(obj_labels)
        obj_pcds_masks = torch.from_numpy(np.array(obj_pcds_masks))

        assert obj_labels.shape[0] == obj_locs.shape[0]
        assert obj_pcds_masks.shape[0] == obj_locs.shape[0]

        return obj_locs, obj_boxes, obj_labels, obj_pcds_masks, scene_pcds


    def _get_pooling_obj_feature(self, args, mv_info_all, sampled_frame_names, scan_id):
        obj_dict = {}
        for i in range(len(sampled_frame_names)):
            frame_info = mv_info_all[sampled_frame_names[i]]
            inst_all = [x for x in frame_info['instance_info'] if x['is_need_process']]
            for one_inst in inst_all:
                tmp_inst_id = one_inst['org_inst_id']
                feat = one_inst[args.inst_feat_type]
                feat = feat[0] if len(feat) == 1 else feat

                inst_id = self.label_converter.orgInstID_to_id[tmp_inst_id]
                if inst_id in obj_dict.keys():
                    obj_dict[inst_id]['feat'].append(feat)
                    assert self.scan_data[scan_id]['inst_labels'][inst_id] == obj_dict[inst_id]['label']
                else:
                    obj_pcd = self.scan_data[scan_id]['obj_pcds'][inst_id]
                    if self.rot_matrix is not None:
                        obj_pcd[:, :3] = np.matmul(obj_pcd[:, :3], self.rot_matrix.transpose())
                    obj_center = obj_pcd[:, :3].mean(0)
                    obj_size = obj_pcd[:, :3].max(0) - obj_pcd[:, :3].min(0)
                    obj_loc = np.concatenate([obj_center, obj_size], 0)

                    obj_box_center = (obj_pcd[:, :3].max(0) + obj_pcd[:, :3].min(0)) / 2
                    obj_box_size = obj_pcd[:, :3].max(0) - obj_pcd[:, :3].min(0)
                    obj_box = np.concatenate([obj_box_center, obj_box_size], 0)

                    obj_dict[inst_id] = {
                        'feat': [feat],
                        'location': obj_loc,
                        'label': self.scan_data[scan_id]['inst_labels'][inst_id],
                        'box' : obj_box,
                    }

        if args.pooling_strategy == 'average_all':
            for key in obj_dict.keys():
                feat_all = np.array(obj_dict[key]['feat'])
                if args.pooling_strategy == 'average_all':
                    obj_dict[key]['feat'] = np.mean(feat_all, axis = 0)

        # for key in obj_dict.keys():
        #     obj_dict[key]['feat'] = list(obj_dict[key]['feat'])
        #     obj_dict[key]['location'] = list(obj_dict[key]['location'])
        #     obj_dict[key]['box'] = list(obj_dict[key]['box'])

        return obj_dict

    def init_dataset_params(self, dataset_cfg):
        if dataset_cfg is None:
            dataset_cfg = {}
        self.pc_type = dataset_cfg.get('pc_type', 'gt')
        self.sem_type = dataset_cfg.get('sem_type', '607')
        self.max_obj_len = dataset_cfg.get('max_obj_len', 80)
        self.num_points = dataset_cfg.get('num_points', 1024)
        self.filter_lang = dataset_cfg.get('filter_lang', False)
        self.rot_aug = dataset_cfg.get('rot_aug', True)
        self.train_duplicate = dataset_cfg.get('train_duplicate', 1)

        self.load_multiview_info = self.cfg.data.get('load_multiview_info', False)
        self.load_mask3d_voxel = self.cfg.data.get('load_mask3d_voxel', False)
        self.process_num = self.cfg.data.get('process_num', 20)
        assert self.pc_type in ['gt', 'pred']
        assert self.sem_type in ['607']

    def init_scan_data(self):
        self.scan_data = self._load_scannet(self.scan_ids, self.pc_type,
                                           load_inst_info=self.split!='test',
                                           load_multiview_info = self.load_multiview_info,
                                           load_mask3d_voxel = self.load_mask3d_voxel,
                                           process_num = self.process_num
                                           )

        # build unique multiple look up
        for scan_id in self.scan_ids:
            inst_labels = self.scan_data[scan_id]['inst_labels']
            self.scan_data[scan_id]['label_count'] = collections.Counter(
                                    [self.label_converter.id_to_scannetid[l] for l in inst_labels])

    def get_scene(self, scan_id, tgt_object_id_list, tgt_object_name_list, sentence):
        if not isinstance(tgt_object_id_list, list):
            tgt_object_id_list = [tgt_object_id_list]
        if not isinstance(tgt_object_name_list, list):
            tgt_object_name_list = [tgt_object_name_list]
        tgt_obj_boxes = [np.concatenate(convert_pc_to_box(self.scan_data[scan_id]["obj_pcds"][i])) for i in tgt_object_id_list]

        # load pcds and labels
        if self.pc_type == 'gt':
            obj_pcds = self.scan_data[scan_id]['obj_pcds'] # N, 6
            obj_labels = self.scan_data[scan_id]['inst_labels'] # N
        elif self.pc_type == 'pred':
            obj_pcds = self.scan_data[scan_id]['obj_pcds_pred']
            obj_labels = self.scan_data[scan_id]['inst_labels_pred']
            # get obj labels by matching
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
        # do not filter for predicted labels, because these labels are not accurate
        excluded_labels = ['wall', 'floor', 'ceiling']
        def keep_obj(i, obj_label):
            if self.pc_type != 'gt' or i in tgt_object_id_list:
                return True
            category = self.int2cat[obj_label]
            if category in excluded_labels:
                return False
            if self.filter_lang and category not in sentence:
                return False
            return True
        selected_obj_idxs = [i for i, obj_label in enumerate(obj_labels) if keep_obj(i, obj_label)]

        # build tgt object id and box
        if self.pc_type == 'gt':
            tgt_object_label_list = [obj_labels[x] for x in tgt_object_id_list]
            tgt_object_id_iou25_list = tgt_object_id_list
            tgt_object_id_iou50_list = tgt_object_id_list
            # for i, _ in enumerate(tgt_object_label_list):
            #     assert self.int2cat[tgt_object_label_list[i]] == tgt_object_name_list[i]
        elif self.pc_type == 'pred':
            tgt_object_label_list = [self.cat2int[x] for x in tgt_object_name_list]
            tgt_object_id_list_matched = []
            tgt_object_id_iou25_list = []
            tgt_object_id_iou50_list = []
            for cur_id in tgt_object_id_list:
                gt_pcd = self.scan_data[scan_id]["obj_pcds"][cur_id]
                gt_center, gt_box_size = convert_pc_to_box(gt_pcd)
                max_iou = -1
                for i in selected_obj_idxs:
                    obj_center, obj_box_size = convert_pc_to_box(obj_pcds[i])
                    iou = eval_ref_one_sample(construct_bbox_corners(obj_center, obj_box_size),
                                            construct_bbox_corners(gt_center, gt_box_size))
                    if iou > max_iou:
                        max_iou = iou
                        tgt_object_id_matched = i
                    # find tgt iou 25
                    if iou >= 0.25:
                        tgt_object_id_iou25_list.append(i)
                    # find tgt iou 50
                    if iou >= 0.5:
                        tgt_object_id_iou50_list.append(i)
                tgt_object_id_list_matched.append(tgt_object_id_matched)
            tgt_object_id_list = tgt_object_id_list_matched

            tgt_object_id_list = list(set(tgt_object_id_list))
            tgt_object_id_iou25_list = list(set(tgt_object_id_iou25_list))
            tgt_object_id_iou50_list = list(set(tgt_object_id_iou50_list))

        # crop objects to max_obj_len
        if self.max_obj_len < len(selected_obj_idxs):
            pre_selected_obj_idxs = selected_obj_idxs
            # select target first
            if len(tgt_object_id_list) > 0:
                selected_obj_idxs = tgt_object_id_list[:]
            selected_obj_idxs.extend(tgt_object_id_iou25_list)
            selected_obj_idxs.extend(tgt_object_id_iou50_list)
            selected_obj_idxs = list(set(selected_obj_idxs))
            # select object with same semantic class with tgt_object
            remained_obj_idx = []
            for i in pre_selected_obj_idxs:
                label = obj_labels[i]
                if i not in selected_obj_idxs:
                    if label in tgt_object_label_list:
                        selected_obj_idxs.append(i)
                    else:
                        remained_obj_idx.append(i)
                if len(selected_obj_idxs) >= self.max_obj_len:
                    break
            if len(selected_obj_idxs) < self.max_obj_len:
                random.shuffle(remained_obj_idx)
                selected_obj_idxs += remained_obj_idx[:(self.max_obj_len - len(selected_obj_idxs))]
            # assert len(selected_obj_idxs) == self.max_obj_len

        # reorganize ids
        tgt_object_id_list = [selected_obj_idxs.index(id) for id in tgt_object_id_list]
        tgt_object_id_iou25_list = [selected_obj_idxs.index(id) for id in tgt_object_id_iou25_list]
        tgt_object_id_iou50_list = [selected_obj_idxs.index(id) for id in tgt_object_id_iou50_list]

        # build unique multiple
        is_multiple = sum([self.scan_data[scan_id]['label_count'][self.label_converter.id_to_scannetid[x]] 
                           for x in tgt_object_label_list]) > 1

        obj_pcds = [obj_pcds[id] for id in selected_obj_idxs]
        obj_labels = [obj_labels[id] for id in selected_obj_idxs]
        obj_fts, obj_locs, obj_boxes, obj_labels = self._obj_processing_post(obj_pcds, obj_labels, is_need_bbox=True, rot_aug=self.rot_aug)

        data_dict = {
            "scan_id": scan_id,
            "tgt_object_id": torch.LongTensor(tgt_object_id_list),
            "tgt_object_label": torch.LongTensor(tgt_object_label_list),
            "tgt_obj_boxes": tgt_obj_boxes, # only use it for evaluation, because it is w/o augmentation.
            "obj_fts": obj_fts,
            "obj_locs": obj_locs,
            "obj_labels": obj_labels,
            "obj_boxes": obj_boxes,
            "tgt_object_id_iou25": torch.LongTensor(tgt_object_id_iou25_list),
            "tgt_object_id_iou50": torch.LongTensor(tgt_object_id_iou50_list), 
            'is_multiple': is_multiple
        }

        if 'multiview_info' in self.scan_data[scan_id]:
            mv_out_dict = self._get_multiview_info(scan_id)
            obj_mv_fts = [mv_out_dict[oid]['feat'] if oid in mv_out_dict else 
                      np.zeros_like(next(iter(mv_out_dict.values()))['feat']) for oid in selected_obj_idxs]
            data_dict['obj_mv_fts'] = torch.from_numpy(np.array(obj_mv_fts)).float()

        if 'mask3d_voxel' in self.scan_data[scan_id]:
            voxel_out_dict = self.scan_data[scan_id]['mask3d_voxel']
            obj_voxel_fts = [voxel_out_dict[id] if id in voxel_out_dict else
                        np.zeros_like(next(iter(voxel_out_dict.values()))) for id in selected_obj_idxs]
            data_dict['obj_voxel_fts'] = torch.from_numpy(np.array(obj_voxel_fts)).float()

        return data_dict