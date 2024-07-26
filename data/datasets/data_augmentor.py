from functools import partial

import math
import numpy as np
import torch


class DataAugmentor(object):
    def __init__(self, cfg, split, **kwargs):
        self.data_augmentor_queue = []
        self.aug_cfg = cfg
        self.kwargs = kwargs
        aug_config_list = self.aug_cfg.aug_list

        self.data_augmentor_queue = []
        if split == 'train':
            for aug in aug_config_list:
                if aug not in self.aug_cfg:
                    continue
                cur_augmentor = partial(getattr(self, aug), config=self.aug_cfg[aug])
                self.data_augmentor_queue.append(cur_augmentor)

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                obj_pcds: (N, 3 + C_in)
                num_points: (1,)
                ...

        Returns:
        """
        aug_dict = self.init_aug(len(data_dict['obj_pcds']))
        for cur_augmentor in self.data_augmentor_queue:
            aug_dict = cur_augmentor(aug_dict=aug_dict)
        data_dict = self.update_data_dict(data_dict, aug_dict)
        return data_dict

    def scene_aug(self, aug_dict, config):
        # scene translation
        if self.check_key(config.translation) and self.check_p(config.translation):
            n = np.zeros((3))
            for i in range(3):
                n[i] = np.random.randn() * config.translation.value[i]
            aug_dict['scene_trans'] = n
        # scene scaling
        if self.check_key(config.scaling) and self.check_p(config.scaling):
            scaling_fac = np.random.rand() \
                        * (config.scaling.value[1] - config.scaling.value[0]) \
                        + config.scaling.value[0]
            aug_dict['scene_scale'] = scaling_fac
        # scene flip
        if self.check_key(config.flip) and self.check_p(config.flip):
            m = np.eye(3)
            flip_type = np.random.choice(4, 1)
            if flip_type == 0:
                # flip x only
                m[0][0] *= -1
            elif flip_type == 1:
                # flip y only
                m[1][1] *= -1
            elif flip_type == 2:
                # flip x+y
                m[0][0] *= -1
                m[1][1] *= -1
            aug_dict['scene_flip'] = m
        # scene rotation
        if self.check_key(config.rotation) and self.check_p(config.rotation):
            if config.rotation.axis_align:
                _r_angles = [0, math.pi/2, math.pi, math.pi*3/2]
                theta_x = np.random.choice(_r_angles) * config.rotation.value[0]
                theta_y = np.random.choice(_r_angles) * config.rotation.value[1]
                theta_z = np.random.choice(_r_angles) * config.rotation.value[2]
            else:
                theta_x = (np.random.rand() * 2 * math.pi - math.pi) * config.rotation.value[0]
                theta_y = (np.random.rand() * 2 * math.pi - math.pi) * config.rotation.value[1]
                theta_z = (np.random.rand() * 2 * math.pi - math.pi) * config.rotation.value[2]
            rx = np.array \
                ([[1, 0, 0],
                  [0, math.cos(theta_x), -math.sin(theta_x)],
                  [0, math.sin(theta_x), math.cos(theta_x)]])
            ry = np.array \
                ([[math.cos(theta_y), 0, math.sin(theta_y)],
                  [0, 1, 0],
                  [-math.sin(theta_y), 0, math.cos(theta_y)]])
            rz = np.array \
                ([[math.cos(theta_z), math.sin(theta_z), 0],
                  [-math.sin(theta_z), math.cos(theta_z), 0],
                  [0, 0, 1]])
            rot_mats = [rx, ry, rz]
            if config.rotation.get('shuffle', False):
                np.random.shuffle(rot_mats)
            r = rot_mats[0].dot(rot_mats[1]).dot(rot_mats[2])
            aug_dict['scene_rot'] = r
        # scene color jitter
        if self.check_key(config.color_jitter):
            rgb_delta = np.random.randn(3) * 0.1
            aug_dict['rgb_delta'] = rgb_delta
        # scene order suffle
        if self.check_key(config.order_shuffle):
            aug_dict['obj_order'] = np.random.permutation(len(aug_dict['obj_order']))

        return aug_dict

    def obj_aug(self, aug_dict, config):
        obj_len = len(aug_dict['obj_order'])
        obj_trans = []
        obj_rot = []
        for _ in range(obj_len):
            n = None
            r = None
            # object translation
            if self.check_key(config.translation) and self.check_p(config.translation):
                n = np.zeros((3))
                for i in range(3):
                    n[i] = np.random.randn() * config.translation.value[i]
            obj_trans.append(n)
            # object rotation
            if self.check_key(config.rotation) and self.check_p(config.rotation):
                if config.rotation.axis_align:
                    _r_angles = [0, math.pi/2, math.pi, math.pi*3/2]
                    theta_x = np.random.choice(_r_angles) * config.rotation.value[0]
                    theta_y = np.random.choice(_r_angles) * config.rotation.value[1]
                    theta_z = np.random.choice(_r_angles) * config.rotation.value[2]
                else:
                    theta_x = (np.random.rand() * 2 * math.pi - math.pi) * config.rotation.value[0]
                    theta_y = (np.random.rand() * 2 * math.pi - math.pi) * config.rotation.value[1]
                    theta_z = (np.random.rand() * 2 * math.pi - math.pi) * config.rotation.value[2]
                rx = np.array \
                    ([[1, 0, 0], 
                      [0, math.cos(theta_x), -math.sin(theta_x)], 
                      [0, math.sin(theta_x), math.cos(theta_x)]])
                ry = np.array \
                    ([[math.cos(theta_y), 0, math.sin(theta_y)],
                      [0, 1, 0],
                      [-math.sin(theta_y), 0, math.cos(theta_y)]])
                rz = np.array \
                    ([[math.cos(theta_z), math.sin(theta_z), 0],
                      [-math.sin(theta_z), math.cos(theta_z), 0],
                      [0, 0, 1]])
                rot_mats = [rx, ry, rz]
                if config.rotation.get('shuffle', False):
                    np.random.shuffle(rot_mats)
                r = rot_mats[0].dot(rot_mats[1]).dot(rot_mats[2])
            obj_rot.append(r)
        aug_dict['obj_trans'] = obj_trans
        aug_dict['obj_rot'] = obj_rot
        # object jitter
        if self.check_key(config.random_jitter):
            aug_dict['obj_jitter'] = config.random_jitter.value
        # object pts shuffle
        if self.check_key(config.pts_shuffle):
            aug_dict['pts_shuffle'] = True
        return aug_dict

    def update_data_dict(self, data_dict, aug_dict):
        data_dict['obj_sizes'] = []
        for i, _ in enumerate(data_dict['obj_pcds']):
            # scene flip
            if aug_dict['scene_flip'] is not None:
                data_dict['obj_pcds'][i][:, :3] = self.rot_fn(data_dict['obj_pcds'][i][:, :3],
                                                               aug_dict['scene_flip'])
            # scene scaling
            if aug_dict['scene_scale'] is not None:
                data_dict['obj_pcds'][i][:, :3] = self.scaling_fn(data_dict['obj_pcds'][i][:, :3],
                                                                  aug_dict['scene_scale'])
            # subsample
            data_dict['obj_pcds'][i] = self.subsample_fn(data_dict['obj_pcds'][i],
                                                         data_dict['num_points'])
            # jitter
            if aug_dict['obj_jitter'] is not None:
                data_dict['obj_pcds'][i][:, :3] = self.jitter_fn(data_dict['obj_pcds'][i][:, :3],
                                                                 aug_dict['obj_jitter'])
            # calc obj size
            data_dict['obj_sizes'].append(data_dict['obj_pcds'][i][:, :3].max(0)
                                        - data_dict['obj_pcds'][i][:, :3].min(0))
            # scene translation
            if aug_dict['scene_trans'] is not None:
                data_dict['obj_pcds'][i][:, :3] += aug_dict['scene_trans']
            # obj translation
            if aug_dict['obj_trans'] and aug_dict['obj_trans'][i] is not None:
                data_dict['obj_pcds'][i][:, :3] += aug_dict['obj_trans'][i]

        #     # scene rotation
        #     if aug_dict['scene_rot'] is not None:
        #         data_dict['obj_pcds'][i][:, :3] = self.rot_fn(data_dict['obj_pcds'][i][:, :3],
        #                                                       aug_dict['scene_rot'])
        #         if 'bg_pcds' in data_dict.keys():
        #             data_dict['bg_pcds'][:, :3] = self.rot_fn(data_dict['bg_pcds'][:, :3],
        #                                                               aug_dict['scene_rot'])

        # scene rotation
        if aug_dict['scene_rot'] is not None:
            data_dict['obj_pcds'] = torch.Tensor(np.array(data_dict['obj_pcds']))
            data_dict['obj_pcds'][:, :, :3] @= aug_dict['scene_rot']
            # data_dict['obj_pcds'][:, :3] = self.rot_fn(data_dict['obj_pcds'][i][:, :3],
            #                                                 aug_dict['scene_rot'])
            if 'bg_pcds' in data_dict.keys():
                data_dict['bg_pcds'][:, :3] @= aug_dict['scene_rot']
                # ['scene_rot']= self.rot_fn(data_dict['bg_pcds'][:, :3],
                #                                             aug_dict['scene_rot'])
        for i, _ in enumerate(data_dict['obj_pcds']):
            # obj rotation
            if aug_dict['obj_rot'] and aug_dict['obj_rot'][i] is not None:
                data_dict['obj_pcds'][i][:, :3] = self.obj_rot_fn(data_dict['obj_pcds'][i][:, :3],
                                                                  aug_dict['obj_rot'][i])
            # scene color jitter
            if aug_dict['rgb_delta'] is not None:
                data_dict['obj_pcds'][i][:, 3:] += aug_dict['rgb_delta']
            # pts shuffle
            if aug_dict['pts_shuffle']:
                data_dict['obj_pcds'][i] = self.pts_shuffle_fn(data_dict['obj_pcds'][i])
        # object order
        data_dict['obj_order'] = aug_dict['obj_order']
        return data_dict

    @staticmethod
    def init_aug(obj_len):
        keys = ['scene_trans', 'scene_flip', 'scene_rot', 'scene_scale', 'rgb_delta',
                'obj_trans', 'obj_rot', 'obj_jitter', 'pts_shuffle']
        aug_dict = {key: None for key in keys}
        aug_dict['obj_order'] = list(np.arange(obj_len))
        return aug_dict

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

    @staticmethod
    def check_p(key):
        return (not isinstance(key, dict)) or ('p' not in key) or (np.random.rand() < key['p'])

    @staticmethod
    def rot_fn(x, mat):
        return np.matmul(x, mat)

    @staticmethod
    def obj_rot_fn(x, mat):
        center = x.mean(0)
        return np.matmul(x - center, mat) + center

    @staticmethod
    def scaling_fn(x, scale):
        center = x.mean(0)
        return (x - center) * scale + center

    @staticmethod
    def jitter_fn(x, scale):
        return x + (np.random.randn(len(x), 3) - 0.5) * scale

    @staticmethod
    def subsample_fn(x, num_points):
        pcd_idxs = np.random.choice(len(x), size=num_points, replace=len(x) < num_points)
        return x[pcd_idxs]

    @staticmethod
    def pts_shuffle_fn(x):
        return x[np.random.permutation(len(x))]
