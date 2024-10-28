import json
from glob import glob
from omegaconf import OmegaConf
from joblib import Parallel, delayed, parallel_backend

import torch
import numpy as np
import trimesh
import open3d as o3d
from tqdm import tqdm

from preprocess.build import ProcessorBase
from preprocess.utils.label_convert import RSCAN_SCANNET as label_convert
from preprocess.utils.align_utils import compute_box_3d, calc_align_matrix, rotate_z_axis_by_degrees
from preprocess.utils.constant import *


class RScanProcessor(ProcessorBase):
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
        scan_ids = [path.split('/')[-1] for path in scan_paths]
        return scan_ids

    def process_point_cloud(self, scan_id, plydata, annotations):
        plylabel, segments, aggregation = annotations
        vertices = plydata.vertices
        vertex_colors = trimesh.visual.uv_to_color(plydata.visual.uv, plydata.visual.material.image)
        vertex_colors = vertex_colors[:, :3] / 255.0

        none_list = list()
        seg_to_inst = {} # segment id to object id
        inst_to_label = {} # object id to label name
        seg_indices = segments['segIndices']
        seg_group = aggregation['segGroups']
        bbox_list = []
        for i, _ in enumerate(seg_group):
            if seg_group[i]['label'] not in label_convert:
                none_list.append(seg_group[i]['label'])
                continue
            inst_to_label[seg_group[i]['id']] = label_convert[seg_group[i]['label']]

            rotation = np.array(seg_group[i]["obb"]["normalizedAxes"]).reshape(3, 3)
            transform = np.array(seg_group[i]["obb"]["centroid"]).reshape(-1, 3)
            scale = np.array(seg_group[i]["obb"]["axesLengths"]).reshape(-1, 3)
            trns = np.eye(4)
            trns[0:3, 3] = transform
            trns[0:3, 0:3] = rotation.T
            box3d = compute_box_3d(scale.reshape(3).tolist(), transform, rotation)
            bbox_list.append(box3d)

            for j in seg_group[i]['segments']:
                seg_to_inst[j] = seg_group[i]['id']
                assert seg_group[i]['id'] == seg_group[i]['objectId']
                assert seg_group[i]['id'] > 0

        query_points = vertices
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(plylabel.vertices, dtype=np.float64))
        tree = o3d.geometry.KDTreeFlann(pcd)

        out_instance = []

        for i, _ in enumerate(query_points):
            point = query_points[i]
            [k, idx, distance] = tree.search_radius_vector_3d(point,0.1)
            if k == 0:
                out_instance.append(-1)
            else:
                nn_idx = idx[0]
                if seg_indices[nn_idx] not in seg_to_inst.keys():
                    out_instance.append(-1)
                else:
                    out_instance.append(seg_to_inst[seg_indices[nn_idx]])

        # alignment: axis-aligned rotation
        align_angle = calc_align_matrix(bbox_list)
        vertices = rotate_z_axis_by_degrees(np.array(vertices), align_angle)
        # alignment: color range
        if np.max(vertex_colors) <= 1:
            vertex_colors = vertex_colors * 255.0
        # alignment: translation
        center_points = np.mean(vertices, axis=0)
        center_points[2] = np.min(vertices[:, 2])
        vertices= vertices - center_points
        vertex_instance = np.array(out_instance)

        assert vertex_colors.shape == vertices.shape
        assert vertex_colors.shape[0] == vertex_instance.shape[0]

        if self.check_key(self.output.pcd):
            torch.save(inst_to_label, self.inst2label_path / f"{scan_id}.pth")
            torch.save((vertices, vertex_colors, vertex_instance), self.pcd_path / f"{scan_id}.pth")
            np.save(self.pcd_path / f"{scan_id}_align_angle.npy", align_angle)

    def scene_proc(self, scan_id):
        data_root = self.data_root / scan_id
        plydata = trimesh.load(data_root / 'mesh.refined.v2.obj', process=False)
        if not (data_root / 'labels.instances.annotated.v2.ply').exists():
            return
        plylabel = trimesh.load(data_root / 'labels.instances.annotated.v2.ply', process=False)
        with open((data_root / 'mesh.refined.0.010000.segs.v2.json'), "r", encoding='utf-8') as f:
            segments = json.load(f)
        with open((data_root / 'semseg.v2.json'), "r", encoding='utf-8') as f:
            aggregation = json.load(f)

        # process point cloud
        self.process_point_cloud(scan_id, plydata, (plylabel, segments, aggregation))

    def process_scans(self):
        scan_ids = self.read_all_scans()
        self.log_starting_info(len(scan_ids))

        if self.num_workers > 1:
            with parallel_backend('multiprocessing', n_jobs=self.num_workers):
                Parallel()(delayed(self.scene_proc)(scan_id) for scan_id in tqdm(scan_ids))
        else:
            for scan_id in tqdm(scan_ids):
                self.scene_proc(scan_id)


if __name__ == '__main__':
    cfg = OmegaConf.create({
        'data_root': '/path/to/3RScan',
        'save_root': '/output/path/to/3RScan',
        'num_workers': 1,
        'output': {
            'pcd': True,
        }
    })
    processor = RScanProcessor(cfg)
    processor.process_scans()
