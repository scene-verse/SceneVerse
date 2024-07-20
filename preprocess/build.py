from pathlib import Path
from fvcore.common.registry import Registry

PROCESSOR_REGISTRY = Registry("Processor")


class ProcessorBase:
    def __init__(self, cfg):
        self.data_root = Path(cfg.data_root)
        self.save_root = Path(cfg.save_root) if cfg.get('save_root', None) else self.data_root.parent / 'scan_data'
        self.num_workers = cfg.num_workers
        self.inst2label_path = self.save_root / 'scan_data' / 'instance_id_to_label'
        self.pcd_path = self.save_root / 'scan_data' / 'pcd_with_global_alignment'
        self.segm_path = self.save_root / 'scan_data' / 'segm'
        self.obj_path = self.save_root / 'scan_data' / 'obj'
        self.sp_path = self.save_root / 'scan_data' / 'super_points'

        self.output = cfg.output

        self.setup_directories()

    def setup_directories(self):
        if self.check_key(self.output.pcd):
            self.inst2label_path.mkdir(parents=True, exist_ok=True)
            self.pcd_path.mkdir(parents=True, exist_ok=True)

    def log_starting_info(self, scan_len, e=''):
        print('='*50)
        print(f'Preprocessing in {self.__class__.__name__} with {scan_len} scans')
        o = [str(i) for i in self.output if i]
        assert len(o) > 0, 'Please specify at least one output type'
        print(f"Output: {', '.join(o)}")
        if len(e) > 0:
            print(e)
        print('='*50)

    @staticmethod
    def check_key(key):
        exist = key is not None
        if not exist:
            return False
        if isinstance(key, bool):
            enabled = key
        elif isinstance(key, dict):
            enabled = key.get('enabled', True)
        elif hasattr(key, 'enabled'):
            enabled = key.get('enabled')
        else:
            enabled = True
        return enabled
