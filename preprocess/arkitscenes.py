import json
from glob import glob
from omegaconf import OmegaConf
from joblib import Parallel, delayed, parallel_backend

import torch
import numpy as np
import trimesh
from tqdm import tqdm
from scipy.spatial.transform import Rotation

from preprocess.build import ProcessorBase
from preprocess.utils.label_convert import ARKITSCENE_SCANNET as label_convert
from preprocess.utils.align_utils import compute_box_3d, calc_align_matrix, rotate_z_axis_by_degrees
from preprocess.utils.constant import *


class ARKitScenesProcessor(ProcessorBase):
    def record_splits(self, scan_ids):
        split_dir = self.save_root / 'split'
        split_dir.mkdir(exist_ok=True)
        if (split_dir / 'train_split.txt').exists() and (split_dir / 'val_split.txt').exists():
            return
        split = {
            'train': [],
            'val': []}
        split['train'] = [scan_id[1] for scan_id in scan_ids if scan_id[0] == 'Training']
        split['val'] = [scan_id[1] for scan_id in scan_ids if scan_id[0] == 'Validation']
        for _s, _c in split.items():
            with open(split_dir / f'{_s}_split.txt', 'w', encoding='utf-8') as fp:
                fp.write('\n'.join(_c))

    def read_all_scans(self):
        scan_ids = []
        for split in ['Training', 'Validation']:
            scan_paths = glob(str(self.data_root) + f'/{split}/*')
            scan_ids.extend([(split, path.split('/')[-1]) for path in scan_paths])
        return scan_ids

    def process_point_cloud(self, scan_id, plydata, annotations):
        vertices = plydata.vertices
        vertex_colors = plydata.visual.vertex_colors
        vertex_colors = vertex_colors[:, :3]

        vertex_instance = np.zeros((vertices.shape[0]))
        inst_to_label = {}
        bbox_list = []

        for _i, label_info in enumerate(annotations["data"]):
            obj_label = label_info["label"]
            object_id = _i + 1
            rotation = np.array(label_info["segments"]["obbAligned"]["normalizedAxes"]).reshape(3, 3)
            r = Rotation.from_matrix(rotation)

            transform = np.array(label_info["segments"]["obbAligned"]["centroid"]).reshape(-1, 3)
            scale = np.array(label_info["segments"]["obbAligned"]["axesLengths"]).reshape(-1, 3)
            trns = np.eye(4)
            trns[0:3, 3] = transform
            trns[0:3, 0:3] = rotation.T
            box_trimesh_fmt = trimesh.creation.box(scale.reshape(3,), trns)
            obj_containment = np.argwhere(box_trimesh_fmt.contains(vertices))

            vertex_instance[obj_containment] = object_id
            inst_to_label[object_id] = label_convert[obj_label]

            box3d = compute_box_3d(scale.reshape(3).tolist(), transform, rotation)
            bbox_list.append(box3d)
        if len(bbox_list) == 0:
            return

        align_angle = calc_align_matrix(bbox_list)
        vertices = rotate_z_axis_by_degrees(np.array(vertices), align_angle)
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
            np.save(self.pcd_path / f"{scan_id}_align_angle.npy", align_angle)

    def scene_proc(self, scan_id):
        split = scan_id[0]
        scan_id = scan_id[1]
        data_root = self.data_root / split / scan_id

        if not (data_root / f'{scan_id}_3dod_mesh.ply').exists():
            return
        if not (data_root / f'{scan_id}_3dod_annotation.json').exists():
            return

        plydata = trimesh.load(data_root / f'{scan_id}_3dod_mesh.ply', process=False)
        with open((data_root / f'{scan_id}_3dod_annotation.json'), "r", encoding='utf-8') as f:
            annotations = json.load(f)

        # process point cloud
        self.process_point_cloud(scan_id, plydata, annotations)

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
        'data_root': '/path/to/ARKitScenes',
        'save_root': '/output/path/to/ARKitScenes',
        'num_workers': 1,
        'output': {
            'pcd': True,
        }
    })
    processor = ARKitScenesProcessor(cfg)
    processor.process_scans()
