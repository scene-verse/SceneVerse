import torch
import torch.nn.functional as F
from torch import nn

from modules.build import HEADS_REGISTRY


class FC(nn.Module):
    def __init__(self, in_size, out_size, pdrop=0., use_gelu=True):
        super(FC, self).__init__()
        self.pdrop = pdrop
        self.use_gelu = use_gelu
        self.linear = nn.Linear(in_size, out_size)
        if use_gelu:
            # self.relu = nn.Relu(inplace=True)
            self.gelu = nn.GELU()
        if pdrop > 0:
            self.dropout = nn.Dropout(pdrop)

    def forward(self, x):
        x = self.linear(x)
        if self.use_gelu:
            # x = self.relu(x)
            x = self.gelu(x)
        if self.pdrop > 0:
            x = self.dropout(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, pdrop=0., use_gelu=True):
        super().__init__()
        self.fc = FC(in_size, mid_size, pdrop=pdrop, use_gelu=use_gelu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))


class AttFlat(nn.Module):
    def __init__(self, hidden_size, flat_mlp_size=512, flat_glimpses=1, flat_out_size=1024, pdrop=0.1):
        super().__init__()
        self.mlp = MLP(
            in_size=hidden_size,
            mid_size=flat_mlp_size,
            out_size=flat_glimpses,
            pdrop=pdrop,
            use_gelu=True
        )
        self.flat_glimpses = flat_glimpses
        self.linear_merge = nn.Linear(
            hidden_size * flat_glimpses,
            flat_out_size
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        if x_mask is not None:
            # att = att.masked_fill(x_mask.squeeze(1).squeeze(1).unsqueeze(2), -1e9)
            att = att.masked_fill(x_mask.unsqueeze(2), -1e9)
        att = F.softmax(att, dim=1)
        att_list = []
        for i in range(self.flat_glimpses):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )
        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)
        return x_atted


@HEADS_REGISTRY.register()
class QAHeadV1(nn.Module):
    def __init__(self, cfg, hidden_size=768, mlp_size=256, glimpse=1, flat_out_size=512, num_answers=8864):
        super().__init__()
        self.attflat_visual = AttFlat(hidden_size, mlp_size, glimpse, flat_out_size, 0.1)
        self.attflat_lang = AttFlat(hidden_size, mlp_size, glimpse, flat_out_size, 0.1)
        self.answer_cls = nn.Sequential(
            nn.Linear(flat_out_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_answers)
        )
        self.fusion_norm = nn.LayerNorm(flat_out_size)

    def forward(self, obj_embeds, obj_masks, txt_embeds, txt_masks, **kwargs):
        object_feat = self.attflat_visual(obj_embeds, obj_masks.logical_not())
        lang_feat = self.attflat_lang(txt_embeds, txt_masks.logical_not())
        fuse_feat = self.fusion_norm(lang_feat + object_feat)
        answer_scores = self.answer_cls(fuse_feat)
        return answer_scores