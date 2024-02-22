from omegaconf import OmegaConf
from torch.utils.data import DataLoader, default_collate, ConcatDataset
from fvcore.common.registry import Registry

from .datasets.dataset_wrapper import DATASETWRAPPER_REGISTRY

DATASET_REGISTRY = Registry("dataset")
DATASET_REGISTRY.__doc__ = """
Registry for datasets, which takes a list of dataset names and returns a dataset object.
Currently it performs similar as registering dataset loading functions, but remains in a
form of object class for future purposes.
"""

def get_dataset(cfg, split):
    assert cfg.data.get(split), f"No valid dataset name in {split}."
    dataset_list = []
    print(split, ': ', ', '.join(cfg.data.get(split)))
    for dataset_name in cfg.data.get(split):
        _dataset = DATASET_REGISTRY.get(dataset_name)(cfg, split)
        assert len(_dataset), f"Dataset '{dataset_name}' is empty!"
        wrapper = cfg.data_wrapper.get(split, cfg.data_wrapper) if not isinstance(cfg.data_wrapper, str) else cfg.data_wrapper
        _dataset = DATASETWRAPPER_REGISTRY.get(wrapper)(cfg, _dataset, split=split)
        # Conduct voxelization
        # TODO: fix voxel config
        if cfg.data.get('use_voxel', None):
            _dataset = DATASETWRAPPER_REGISTRY.get('VoxelDatasetWrapper')(cfg, _dataset)
        dataset_list.append(_dataset)

    print('='*50)
    print('Dataset\t\t\tSize')
    total = sum([len(dataset) for dataset in dataset_list])
    for dataset_name, dataset in zip(cfg.data.get(split), dataset_list):
        print(f'{dataset_name:<20} {len(dataset):>6} ({len(dataset) / total * 100:.1f}%)')
    print(f'Total\t\t\t{total}')
    print('='*50)
    if split == 'train':
        dataset_list = ConcatDataset(dataset_list)

    return dataset_list


def build_dataloader(cfg, split='train'):
    """_summary_
    Unittest:
        dataloader_train = build_dataloader(default_cfg, split='train')
        for _item in dataloader_train:
            print(_item.keys())

    Args:
        cfg (_type_): _description_
        split (str, optional): _description_. Defaults to 'train'.

    Returns:
        _type_: _description_
    """
    if split == 'train':
        dataset = get_dataset(cfg, split)
        return DataLoader(dataset,
                          batch_size=cfg.dataloader.batchsize,
                          num_workers=cfg.dataloader.num_workers,
                          collate_fn=getattr(dataset.datasets[0], 'collate_fn', default_collate),
                          pin_memory=True, # TODO: Test speed
                          prefetch_factor=2 if not cfg.debug.flag else None,
                          persistent_workers=True if not cfg.debug.flag else None,
                          shuffle=True,
                          drop_last=True)
    else:
        loader_list = []
        for dataset in get_dataset(cfg, split):
            loader_list.append(
                DataLoader(dataset,
                    batch_size=cfg.dataloader.get('batchsize_eval', cfg.dataloader.batchsize),
                    num_workers=cfg.dataloader.num_workers,
                    collate_fn=getattr(dataset, 'collate_fn', default_collate),
                    pin_memory=True, # TODO: Test speed
                    prefetch_factor=2 if not cfg.debug.flag else None,
                    persistent_workers=True if not cfg.debug.flag else None,
                    shuffle=False))
        # TODO: temporary solution for backward compatibility.
        if len(loader_list) == 1:
            return loader_list[0]
        else:
            return loader_list


if __name__ == '__main__':
    pass
