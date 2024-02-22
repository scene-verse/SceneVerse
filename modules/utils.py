import copy

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F


#########################################################
# General modules helpers
#########################################################
def get_activation_fn(activation_type):
    if activation_type not in ["relu", "gelu", "glu"]:
        raise RuntimeError(f"activation function currently support relu/gelu, not {activation_type}")
    return getattr(F, activation_type)


def get_mlp_head(input_size, hidden_size, output_size, dropout=0):
    return nn.Sequential(*[
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.LayerNorm(hidden_size, eps=1e-12),
        nn.Dropout(dropout),
        nn.Linear(hidden_size, output_size)
    ])


def layer_repeat(module, N, share_layer=False):
    if share_layer:
        return nn.ModuleList([module] * N)
    else:
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N - 1)] + [module])


#########################################################
# Specific modules helpers
#########################################################
def calc_pairwise_locs(obj_centers, obj_whls, eps=1e-10, pairwise_rel_type='center', spatial_dist_norm=True,
                       spatial_dim=5):
    if pairwise_rel_type == 'mlp':
        obj_locs = torch.cat([obj_centers, obj_whls], 2)
        pairwise_locs = torch.cat(
            [einops.repeat(obj_locs, 'b l d -> b l x d', x=obj_locs.size(1)),
             einops.repeat(obj_locs, 'b l d -> b x l d', x=obj_locs.size(1))],
            dim=3
        )
        return pairwise_locs

    pairwise_locs = einops.repeat(obj_centers, 'b l d -> b l 1 d') \
                    - einops.repeat(obj_centers, 'b l d -> b 1 l d')
    pairwise_dists = torch.sqrt(torch.sum(pairwise_locs ** 2, 3) + eps)  # (b, l, l)
    if spatial_dist_norm:
        max_dists = torch.max(pairwise_dists.view(pairwise_dists.size(0), -1), dim=1)[0]
        norm_pairwise_dists = pairwise_dists / einops.repeat(max_dists, 'b -> b 1 1')
    else:
        norm_pairwise_dists = pairwise_dists

    if spatial_dim == 1:
        return norm_pairwise_dists.unsqueeze(3)

    pairwise_dists_2d = torch.sqrt(torch.sum(pairwise_locs[..., :2] ** 2, 3) + eps)
    if pairwise_rel_type == 'center':
        pairwise_locs = torch.stack(
            [norm_pairwise_dists, pairwise_locs[..., 2] / pairwise_dists,
             pairwise_dists_2d / pairwise_dists, pairwise_locs[..., 1] / pairwise_dists_2d,
             pairwise_locs[..., 0] / pairwise_dists_2d],
            dim=3
        )
    elif pairwise_rel_type == 'vertical_bottom':
        bottom_centers = torch.clone(obj_centers)
        bottom_centers[:, :, 2] -= obj_whls[:, :, 2]
        bottom_pairwise_locs = einops.repeat(bottom_centers, 'b l d -> b l 1 d') \
                               - einops.repeat(bottom_centers, 'b l d -> b 1 l d')
        bottom_pairwise_dists = torch.sqrt(torch.sum(bottom_pairwise_locs ** 2, 3) + eps)  # (b, l, l)
        bottom_pairwise_dists_2d = torch.sqrt(torch.sum(bottom_pairwise_locs[..., :2] ** 2, 3) + eps)
        pairwise_locs = torch.stack(
            [norm_pairwise_dists,
             bottom_pairwise_locs[..., 2] / bottom_pairwise_dists,
             bottom_pairwise_dists_2d / bottom_pairwise_dists,
             pairwise_locs[..., 1] / pairwise_dists_2d,
             pairwise_locs[..., 0] / pairwise_dists_2d],
            dim=3
        )

    if spatial_dim == 4:
        pairwise_locs = pairwise_locs[..., 1:]
    return pairwise_locs

def calc_pairwise_locs_mv(obj_centers, pairwise_rel_type='center', spatial_dist_norm=True, spatial_dim=5):
    eps=1e-10
    pairwise_locs = einops.repeat(obj_centers, 'b l d -> b l 1 d') \
                    - einops.repeat(obj_centers, 'b l d -> b 1 l d')
    pairwise_dists = torch.sqrt(torch.sum(pairwise_locs ** 2, 3) + eps)  # (b, l, l)
    if spatial_dist_norm:
        max_dists = torch.max(pairwise_dists.view(pairwise_dists.size(0), -1), dim=1)[0]
        norm_pairwise_dists = pairwise_dists / einops.repeat(max_dists, 'b -> b 1 1')
    else:
        norm_pairwise_dists = pairwise_dists

    if spatial_dim == 1:
        return norm_pairwise_dists.unsqueeze(3)

    pairwise_dists_2d = torch.sqrt(torch.sum(pairwise_locs[..., :2] ** 2, 3) + eps)
    if pairwise_rel_type == 'center':
        pairwise_locs = torch.stack(
            [norm_pairwise_dists, pairwise_locs[..., 2] / pairwise_dists,
             pairwise_dists_2d / pairwise_dists, pairwise_locs[..., 1] / pairwise_dists_2d,
             pairwise_locs[..., 0] / pairwise_dists_2d],
            dim=3
        )

    if spatial_dim == 4:
        pairwise_locs = pairwise_locs[..., 1:]
    return pairwise_locs

# TODO: need to generalize this function to more use cases to be in modules/utils.py
def get_mixup_function(mixup_strategy, mixup_stage1, mixup_stage2):
    if mixup_strategy is None:
        return None
    assert mixup_strategy in ['linear_decay', 'all_mixup']

    if mixup_strategy == 'linear_decay':
        return LinearDecayMixup(mixup_stage1, mixup_stage2)
    elif mixup_strategy == 'all_mixup':
        return AllMixup()


class AllMixup(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, obj_sem_cls_pred, obj_labels, cur_step, total_steps):
        mixup_sem_cls_pred = torch.zeros_like(obj_sem_cls_pred)
        for i in range(mixup_sem_cls_pred.shape[0]):
            for j in range(mixup_sem_cls_pred.shape[1]):
                if obj_labels[i, j] >= 0:
                    mixup_sem_cls_pred[i, j, obj_labels[i, j]] = 1.0
        return mixup_sem_cls_pred


class LinearDecayMixup(nn.Module):
    def __init__(self, mixup_stage1, mixup_stage2) -> None:
        super().__init__()
        self.stage1_rate = mixup_stage1
        self.stage2_rate = mixup_stage2
        assert self.stage2_rate > self.stage1_rate

    def forward(self, obj_sem_cls_pred, obj_labels, cur_step, total_steps):
        if cur_step < total_steps * self.stage1_rate:
            mixup_ratio = 1.0
        elif cur_step < total_steps * self.stage2_rate:
            mixup_ratio = (total_steps * self.stage2_rate - cur_step) / (
                        (self.stage2_rate - self.stage1_rate) * total_steps)
        else:
            mixup_ratio = 0.0
        # mixup
        mixup_sem_cls_pred = obj_sem_cls_pred.clone()  # B, O, 607
        random_numer = torch.rand(mixup_sem_cls_pred.shape[0:2])  # B, O
        mixup_mask = random_numer < mixup_ratio
        for i in range(mixup_sem_cls_pred.shape[0]):
            for j in range(mixup_sem_cls_pred.shape[1]):
                if mixup_mask[i, j] and obj_labels[i, j] >= 0:
                    mixup_sem_cls_pred[i, j, :] = 0.0
                    mixup_sem_cls_pred[i, j, obj_labels[i, j]] = 1.0
        return mixup_sem_cls_pred