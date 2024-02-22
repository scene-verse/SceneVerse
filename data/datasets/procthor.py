import collections

from ..build import DATASET_REGISTRY
from .base import ScanBase


@DATASET_REGISTRY.register()
class ProcThorPretrainObj(ScanBase):
    def __init__(self, cfg, split):
        super(ProcThorPretrainObj, self).__init__(cfg, split)
        self.base_dir = cfg.data.procthor_base

        self.load_scene_pcds = cfg.data.args.get('load_scene_pcds', False)
        if self.load_scene_pcds:
            self.max_pcd_num_points = cfg.data.args.get('max_pcd_num_points', None)
            assert self.max_pcd_num_points is not None
        self.bg_points_num = cfg.data.args.get('bg_points_num', 1000)

        self.scan_ids = sorted(list(self._load_split(self.split)))
        if cfg.debug.flag and cfg.debug.debug_size != -1:
            self.scan_ids = self.scan_ids[:cfg.debug.debug_size]

        print(f"Loading ProcThor {split}-set scans")
        self.scan_data = self._load_scan(self.scan_ids)
        self.scan_ids = sorted(list(self.scan_data.keys()))
        print(f"Finish loading ProcThor {split}-set scans of length {len(self.scan_ids)}")

    def __len__(self):
        return len(self.scan_ids)

    def __getitem__(self, index):
        """Data dict post-processing, for example, filtering, crop, nomalization,
        rotation, etc.

        Args:
            index (int): _description_
        """
        data_dict = self._getitem_obj_pretrain(index)
        dataset = 'procthor'
        data_dict['source'] = dataset
        return data_dict


@DATASET_REGISTRY.register()
class ProcThorSpatialRefer(ScanBase):
    def __init__(self, cfg, split):
        super(ProcThorSpatialRefer, self).__init__(cfg, split)
        self.base_dir = cfg.data.procthor_base
        self.max_obj_len = cfg.data.args.max_obj_len - 1
        self.filter_lang = cfg.data.args.filter_lang

        self.load_scene_pcds = cfg.data.args.get('load_scene_pcds', False)
        if self.load_scene_pcds:
            self.max_pcd_num_points = cfg.data.args.get('max_pcd_num_points', None)
            assert self.max_pcd_num_points is not None
        self.bg_points_num = cfg.data.args.get('bg_points_num', 1000)

        sources = cfg.data.get(self.__class__.__name__).get(split).sources
        all_scan_ids = self._load_split(self.split)

        print(f"Loading ProcThor {split}-set language")
        self.lang_data, self.scan_ids = self._load_lang(cfg, all_scan_ids, sources)
        print(f"Finish loading ProcThor {split}-set language of size {self.__len__()}")

        print(f"Loading ProcThor {split}-set scans")
        self.scan_data = self._load_scan(self.scan_ids)
        print(f"Finish loading ProcThor {split}-set scans")

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
