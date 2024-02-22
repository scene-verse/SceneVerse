import torch
import torch.nn as nn

from modules.build import HEADS_REGISTRY
from modules.utils import get_activation_fn


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, hidden_size, hidden_act='gelu'):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.transform_act_fn = get_activation_fn(hidden_act)
        self.LayerNorm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.transform = BertPredictionHeadTransform(hidden_size=hidden_size, hidden_act='gelu')
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


@HEADS_REGISTRY.register()
class PretrainHeadV1(nn.Module):
    def __init__(self, cfg, hidden_size=768, vocab_size=30522):
        super().__init__()
        self.lm_pred_head = BertLMPredictionHead(hidden_size, vocab_size)

    def forward(self, txt_embeds, **kwargs):
        txt_lm_cls_logits = self.lm_pred_head(txt_embeds)
        return txt_lm_cls_logits


@HEADS_REGISTRY.register()
class OVPretrainHead(nn.Module):
    def __init__(self, cfg, hidden_size=768, vocab_size=30522, obj_vocab_size=607):
        super().__init__()
        self.lm_pred_head = BertLMPredictionHead(hidden_size, vocab_size)
        self.obj_pred_head = BertLMPredictionHead(hidden_size, obj_vocab_size)

    def forward(self, txt_embeds, obj_embeds, **kwargs):
        txt_lm_cls_logits = self.lm_pred_head(txt_embeds)
        obj_lm_cls_logits = self.obj_pred_head(obj_embeds)
        return (txt_lm_cls_logits, obj_lm_cls_logits)