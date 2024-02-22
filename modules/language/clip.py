from contextlib import nullcontext
import torch
import torch.nn as nn
from transformers import CLIPTextModelWithProjection

from modules.build import LANGUAGE_REGISTRY
from modules.utils import get_mlp_head

@LANGUAGE_REGISTRY.register()
class CLIPLanguageEncoder(nn.Module):
    def __init__(self, cfg, weights="openai/clip-vit-large-patch14", output_dim=768, freeze_backbone=True, use_projection=False, dropout=0.1):
        super().__init__()
        self.context = torch.no_grad if freeze_backbone else nullcontext
        self.model = CLIPTextModelWithProjection.from_pretrained(weights)
        self.use_projection = use_projection
        if use_projection:
            self.projection = get_mlp_head(self.model.config.hidden_size, output_dim, output_dim, dropout=dropout)
        #self.attention = nn.MultiheadAttention(embed_dim=768, num_heads=12, batch_first=True)
        
    def forward(self, txt_ids, txt_masks):
        with self.context():
            txt = self.model(txt_ids, txt_masks).last_hidden_state
            txt = self.model.text_projection(txt)
            txt = torch.nn.functional.normalize(txt, p=2, dim=2)
        #txt = self.attention(txt, txt, txt, key_padding_mask=txt_masks.logical_not())[0]
        if self.use_projection:
            txt = self.projection(txt)
        return txt