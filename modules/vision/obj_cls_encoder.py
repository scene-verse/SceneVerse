import torch.nn as nn
from modules.build import VISION_REGISTRY
from modules.utils import get_mlp_head

@VISION_REGISTRY.register()
class ObjClsEncoder(nn.Module):
    def __init__(self, cfg, input_feat_size=768, hidden_size=768, tgt_cls_num=607):
        super().__init__()
        self.cfg = cfg
        self.vis_cls_head = get_mlp_head(input_feat_size, hidden_size // 2, tgt_cls_num, dropout = 0.3)

    def forward(self, obj_feats, **kwargs):
        obj_logits = self.vis_cls_head(obj_feats)
        return obj_logits
