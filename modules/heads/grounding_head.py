import torch.nn as nn

from modules.build import HEADS_REGISTRY
from modules.utils import get_mlp_head


@HEADS_REGISTRY.register()
class GroundHeadV1(nn.Module):
    def __init__(self, cfg, input_size=768, hidden_size=768, sem_cls_size=607, dropout=0.3, detach_all_aux_loss=False):
        super().__init__()
        self.og3d_head = get_mlp_head(
            input_size, hidden_size,
            1, dropout=dropout
        )
        self.txt_clf_head = get_mlp_head(
            input_size, hidden_size,
            sem_cls_size, dropout=dropout
        )
        self.obj3d_clf_head = get_mlp_head(
            input_size, hidden_size,
            sem_cls_size, dropout=dropout
        )
        self.obj3d_clf_pre_head = get_mlp_head(
            input_size, hidden_size,
            sem_cls_size, dropout=dropout
        )
        self.detach_all_aux_loss = detach_all_aux_loss

    def forward(self, txt_embeds, obj_embeds, obj_pre_embeds, obj_masks, **kwargs):
        og3d_logits = self.og3d_head(obj_embeds).squeeze(2)
        og3d_logits = og3d_logits.masked_fill_(obj_masks.logical_not(), -float('inf'))
        if self.detach_all_aux_loss:
            txt_embeds = txt_embeds.detach()
            obj_embeds = obj_embeds.detach()
            obj_pre_embeds = obj_pre_embeds.detach()
        txt_cls_logits = self.txt_clf_head(txt_embeds[:, 0])
        obj_cls_logits = self.obj3d_clf_head(obj_embeds)
        obj_cls_pre_logits = self.obj3d_clf_pre_head(obj_pre_embeds)
        return txt_cls_logits, obj_cls_logits, obj_cls_pre_logits, og3d_logits


@HEADS_REGISTRY.register()
class GroundHead(nn.Module):
    def __init__(self, cfg, input_size=768, hidden_size=768, dropout=0.3):
        super().__init__()
        self.og3d_head = get_mlp_head(
            input_size, hidden_size,
            1, dropout=dropout
        )

    def forward(self, obj_embeds, obj_masks=None, **kwargs):
        og3d_logits = self.og3d_head(obj_embeds).squeeze(2)
        if obj_masks is not None:
            og3d_logits = og3d_logits.masked_fill_(obj_masks.logical_not(), -float('inf'))
        return og3d_logits
