import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from optim.loss.loss import LOSS_REGISTRY
from einops import rearrange, einsum

from common.dist_utils import all_gather


@LOSS_REGISTRY.register()
class TextObjWithinBatch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.distributed = cfg.num_gpu > 1
        self.bce = cfg.task in ["ScanQA"]

    def forward(self, data_dict):
        obj_feats = data_dict["intra_obj_embeds"]     # B, O, D
        text_feats = data_dict["intra_text_embed"]    # B, D, feature of CLS token
        labels = data_dict["tgt_object_id"]            # B, 1
        masks = data_dict["obj_masks"]

        # B*L vs. B in per-scene scenario
        if obj_feats.shape[0] != masks.shape[0]:
            masks = masks.unsqueeze(1).repeat(1, int(obj_feats.shape[0] / masks.shape[0]), 1).view(-1, masks.shape[1])
            labels = labels.view(-1, 1)

        obj_feats = F.normalize(obj_feats, dim=-1, p=2)
        text_feats = F.normalize(text_feats, dim=-1, p=2)
        obj2text_logits = einsum(obj_feats, text_feats, "b o d, b d -> b o")
        obj2text_logits = obj2text_logits
        labels = labels.squeeze(-1)
        if self.bce:
            loss = F.binary_cross_entropy_with_logits(obj2text_logits, labels.float(), reduction="sum", weight=masks) / float(labels.shape[0])
        else:
            obj2text_logits.masked_fill_(masks.logical_not(), -float('inf'))
            loss = F.cross_entropy(obj2text_logits, labels)
        return loss


@LOSS_REGISTRY.register()
class TextObjBetweenBatch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.distributed = cfg.num_gpu > 1
        self.logit_scale = nn.Parameter((torch.ones([]) * np.log(1 / 0.07)).exp())

    def forward(self, data_dict):
        logit_scale = torch.clamp(self.logit_scale, max=100)
        obj_feats = data_dict["inter_obj_embeds"]     # B, O, D
        text_feats = data_dict["inter_text_embed"]    # B, D, feature of CLS token
        labels = data_dict["tgt_object_id"]            # B, 1

        if obj_feats.shape[0] != labels.shape[0]:
            labels = labels.view(-1, 1)

        tgt_obj_feats = obj_feats[torch.arange(labels.size(0)), labels[:, 0], :]       # B, D
        tgt_obj_feats = F.normalize(tgt_obj_feats, dim=-1, p=2)
        text_feats = F.normalize(text_feats, dim=-1, p=2)
        if self.distributed:
            tgt_obj_feats, text_feats = all_gather([
                tgt_obj_feats, text_feats
            ])
        pseudo_labels = torch.arange(text_feats.shape[0]).to(text_feats.device)  # B,
        text2obj_logits = logit_scale * text_feats @ tgt_obj_feats.t()  # B, B
        obj2text_logits = logit_scale * tgt_obj_feats @ text_feats.t()  # B, B
        t2o = F.cross_entropy(text2obj_logits, pseudo_labels)
        o2t = F.cross_entropy(obj2text_logits, pseudo_labels)
        loss = (t2o + o2t) / 2
        return loss


@LOSS_REGISTRY.register()
class TextSceneBetweenBatch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.distributed = cfg.num_gpu > 1
        self.logit_scale = nn.Parameter((torch.ones([]) * np.log(1 / 0.07)).exp())

    def forward(self, data_dict):
        logit_scale = torch.clamp(self.logit_scale, max=100)
        scene_feats = data_dict["scene_embed"]     # B, O, D
        text_feats = data_dict["scene_text_embed"]    # B, D, feature of CLS token

        scene_feats = F.normalize(scene_feats, dim=-1, p=2)
        text_feats = F.normalize(text_feats, dim=-1, p=2)
        if self.distributed:
            scene_feats, text_feats = all_gather([
                scene_feats, text_feats
            ])
        pseudo_labels = torch.arange(text_feats.shape[0]).to(text_feats.device)  # B,
        text2scene_logits = logit_scale * text_feats @ scene_feats.t()  # B, B
        scene2text_logits = logit_scale * scene_feats @ text_feats.t()  # B, B
        t2s = F.cross_entropy(text2scene_logits, pseudo_labels)
        s2t = F.cross_entropy(scene2text_logits, pseudo_labels)
        loss = (t2s + s2t) / 2
        return loss


if __name__ == "__main__":
    B, O, D = 32, 10, 512
    data_dict = {
        "obj_embeds": torch.randn(B, O, D),
        "text_embed": torch.randn(B, D),
        "labels": torch.randint(0, O, (B, 1)),
        "obj_masks": torch.ones(B, O).bool(),
    }
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({"num_gpu": 1})
    text2obj_loss = TextObjWithinBatch(cfg)(data_dict)
    obj2text_loss = TextObjBetweenBatch(cfg)(data_dict)

        