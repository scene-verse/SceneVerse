import re
import json
from glob import glob
from omegaconf import OmegaConf
from joblib import Parallel, delayed, parallel_backend

import torch
from plyfile import PlyData
import numpy as np
import pandas as pd
from tqdm import tqdm

from preprocess.build import ProcessorBase
from preprocess.utils.label_convert import MULTISCAN_SCANNET as label_convert
from preprocess.utils.constant import *


class MultiScanProcessor(ProcessorBase):
    def record_splits(self, scan_ids, ratio=0.8):
        split_dir = self.save_root / 'split'
        split_dir.mkdir(exist_ok=True)
        if (split_dir / 'train_split.txt').exists() and (split_dir / 'val_split.txt').exists():
            return
        scan_len = len(scan_ids)
        split = {
            'train': [],
            'val': []}
        cur_split = 'train'
        for scan_id in tqdm(sorted(scan_ids)):
            split[cur_split].append(scan_id)
            if len(split['train']) > ratio*scan_len:
                cur_split = 'val'
        for _s, _c in split.items():
            with open(split_dir / f'{_s}_split.txt', 'w', encoding='utf-8') as fp:
                fp.write('\n'.join(_c))

    def read_all_scans(self):
        scan_paths = glob(str(self.data_root) + '/*')
        scans_df = []
        for scan_path in scan_paths:
            scan_id = re.findall(r"scene\_[0-9]{5}\_[0-9]{2}", scan_path)[0]
            scene_id = '_'.join(scan_id.split('_')[:-1])
            row = pd.DataFrame([[scene_id, scan_id, scan_path]],
                                columns=['sceneId', 'scanId', 'scanPath'])
            scans_df.append(row)
        scans_df = pd.concat(scans_df)
        return scans_df

    def process_point_cloud(self, scan_id, plydata, annotations):
        inst_to_label = {}
        _x = np.asarray(plydata['vertex']['x'])
        _y = np.asarray(plydata['vertex']['y'])
        _z = np.asarray(plydata['vertex']['z'])
        _nx = np.asarray(plydata['vertex']['nx'])
        _ny = np.asarray(plydata['vertex']['ny'])
        _nz = np.asarray(plydata['vertex']['nz'])
        _red = plydata['vertex']['red'].astype('float64')
        _green = plydata['vertex']['green'].astype('float64')
        _blue = plydata['vertex']['blue'].astype('float64')

        vertices = np.column_stack((_x, _y, _z))
        vertex_colors = np.column_stack((_red, _green, _blue))
        vertex_instance = np.zeros((vertices.shape[0]))
        triangles = np.vstack(plydata['face'].data['vertex_indices'])

        object_ids = plydata['face'].data['objectId']
        part_ids = plydata['face'].data['partId']
        semseg_df = pd.DataFrame({'objectId': object_ids, 'partId': part_ids})

        df = self.annotations_to_dataframe_obj(annotations)
        for _, row in df.iterrows():
            object_id = row['objectId']
            assert object_id > 0, f"object id should be greater than 0, but got {object_id}"
            object_label = row['objectLabel'].split('.')[0]
            object_label_sn607 = label_convert[object_label]

            condition1 = semseg_df['objectId'] == object_id
            tri_indices = semseg_df[condition1].index.values
            object_vertices = np.unique(triangles[tri_indices])
            vertex_instance[object_vertices] = object_id
            inst_to_label[object_id] = object_label_sn607

        if np.max(vertex_colors) <= 1:
            vertex_colors = vertex_colors * 255.0
        center_points = np.mean(vertices, axis=0)
        center_points[2] = np.min(vertices[:, 2])
        vertices = vertices - center_points
        assert vertex_colors.shape == vertices.shape
        assert vertex_colors.shape[0] == vertex_instance.shape[0]

        if self.check_key(self.output.pcd):
            torch.save(inst_to_label, self.inst2label_path / f"{scan_id}.pth")
            torch.save((vertices, vertex_colors, vertex_instance), self.pcd_path / f"{scan_id}.pth")

    @staticmethod
    def annotations_to_dataframe_obj(annotations):
        objects = annotations['objects']
        df_list = []
        for obj in objects:
            object_id = obj['objectId']
            object_label = obj['label']
            df_row = pd.DataFrame(
                [[object_id, object_label]],
                columns=['objectId', 'objectLabel']
            )
            df_list.append(df_row)
        df = pd.concat(df_list)
        return df

    def scene_proc(self, scan_id):
        data_root = self.data_root / scan_id
        plydata = PlyData.read(data_root / f'{scan_id}.ply')
        with open((data_root / f'{scan_id}.annotations.json'), "r", encoding='utf-8') as f:
            annotations = json.load(f)

        # process point cloud
        self.process_point_cloud(scan_id, plydata, annotations)

    def process_scans(self):
        scans_df = self.read_all_scans()
        scan_ids = scans_df['scanId'].unique()
        self.log_starting_info(len(scan_ids))

        if self.num_workers > 1:
            with parallel_backend('multiprocessing', n_jobs=self.num_workers):
                Parallel()(delayed(self.scene_proc)(scan_id) for scan_id in tqdm(scan_ids))
        else:
            for scan_id in tqdm(scan_ids):
                print(scan_id)
                self.scene_proc(scan_id)


if __name__ == '__main__':
    cfg = OmegaConf.create({
        'data_root': '/path/to/MultiScan',
        'save_root': '/output/path/to/MultiScan',
        'num_workers': 1,
        'output': {
            'pcd': True,
        }
    })
    processor = MultiScanProcessor(cfg)
    processor.process_scans()
