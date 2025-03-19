import pickle
from glob import glob
from omegaconf import OmegaConf
from joblib import Parallel, delayed, parallel_backend

import torch
import numpy as np
from tqdm import tqdm

from preprocess.build import ProcessorBase
from preprocess.utils.label_convert import S3D_SCANNET as label_convert
from preprocess.utils.constant import *


PTS_LIMIT = 480000


class S3DProcessor(ProcessorBase):
    def record_splits(self, scan_ids):
        split_dir = self.save_root / 'split'
        split_dir.mkdir(exist_ok=True)
        split = {
            'train': [],
            'val': [],
            'test': []}
        split['train'] = [scan_id[1] for scan_id in scan_ids if scan_id[0] == 'train']
        split['val'] = [scan_id[1] for scan_id in scan_ids if scan_id[0] == 'val']
        split['test'] = [scan_id[1] for scan_id in scan_ids if scan_id[0] == 'test']
        for _s, _c in split.items():
            with open(split_dir / f'{_s}_split.txt', 'w', encoding='utf-8') as fp:
                fp.write('\n'.join(_c))

    def read_all_scans(self):
        scan_ids = []
        for split in ['train', 'val', 'test']:
            scan_paths = glob(str(self.data_root) + f'/{split}/*')
            scan_ids.extend([(split, '_'.join(path.split('/')[-1].split('_')[:-2])) for path in scan_paths])
        return scan_ids

    def process_point_cloud(self, scan_id, plydata, annotations):
        vertices = plydata[0]
        vertex_colors = (plydata[1][:,:3] + 1) / 2.0 * 255.0

        vertex_instance = - np.ones((vertices.shape[0]))
        inst_to_label = {}

        for _id, _box in enumerate(annotations['gt_boxes_upright_depth']):
            if annotations['class'][_id] in [38, 39, 40]:
                continue
            centroid = _box[:3]
            dimension = _box[3:6]
            box_max = centroid + dimension/2
            box_min = centroid - dimension/2
            point_max_mask = np.all(vertices < box_max, axis=1)
            point_min_mask = np.all(vertices > box_min, axis=1)
            point_mask = np.logical_and(point_max_mask, point_min_mask)
            vertex_instance[point_mask] = _id
            inst_to_label[_id] = label_convert[annotations['class'][_id]]

        center_points = np.mean(vertices, axis=0)
        center_points[2] = np.min(vertices[:, 2])
        vertices = vertices - center_points
        assert vertex_colors.shape == vertices.shape
        assert vertex_colors.shape[0] == vertex_instance.shape[0]

        if vertices.shape[0] > PTS_LIMIT:
            pcd_idxs = np.random.choice(vertices.shape[0], size=PTS_LIMIT, replace=False)
            vertices = vertices[pcd_idxs]
            colors = colors[pcd_idxs]
            vertex_instance = vertex_instance[pcd_idxs]

        if self.check_key(self.output.pcd):
            torch.save(inst_to_label, self.inst2label_path / f"{scan_id}.pth")
            torch.save((vertices, vertex_colors, vertex_instance), self.pcd_path / f"{scan_id}.pth")

    def scene_proc(self, scan_id):
        split = scan_id[0]
        scan_id = scan_id[1]
        data_root = self.data_root / split

        if not (data_root / f'{scan_id}_1cm_seg.pth').exists():
            return
        if not (self.data_root.parent / 'anno_mask' / f'{scan_id}_1cm.bin').exists():
            return
        plydata = torch.load(data_root / f'{scan_id}_1cm_seg.pth')
        with open(self.data_root.parent / 'anno_mask' / f'{scan_id}_1cm.bin', 'rb') as f:
            annotations = pickle.load(f)

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
    # we use the data processing for Structured3D from Swin3D,
    # please refer to https://github.com/yuxiaoguo/Uni3DScenes for more details.
    cfg = OmegaConf.create({
        'data_root': '/path/to/Structured3D/data_out/swin3d_new',
        'save_root': '/output/path/to/Structured3D',
        'num_workers': 1,
        'output': {
            'pcd': True,
        }
    })
    processor = S3DProcessor(cfg)
    processor.process_scans()
