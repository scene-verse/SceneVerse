import torch
import torch.nn as nn
import json
from pathlib import Path

import clip
from transformers import BertConfig, BertModel, BertTokenizer
from einops import rearrange

from model.build import MODEL_REGISTRY, BaseModel
from modules.layers.pointnet import PointNetPP
from modules.utils import get_mlp_head
from optim.utils import no_decay_param_group


@MODEL_REGISTRY.register()
class ObjCls(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.model_name = cfg.model.get("model_name", "pointnext")
        self.language_type = cfg.model.get("language_type", "clip")
        self.pre_extract_path = cfg.model.get("pre_extract_path", None)

        cls_in_channel = 512 if self.language_type == "clip" else 768
        self.point_feature_extractor = PointNetPP(
            sa_n_points=[32, 16, None],
            sa_n_samples=[32, 32, None],
            sa_radii=[0.2, 0.4, None],
            sa_mlps=[[3, 64, 64, 128], [128, 128, 128, 256], [256, 256, 512, cls_in_channel]],
        )

        if cfg.num_gpu > 1:
            self.point_feature_extractor = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.point_feature_extractor)

        if not cfg.model.open_vocab:
            cls_hidden = cfg.model.get("cls_hidden", 1024)
            num_classes = cfg.model.num_classes
            self.cls_head = get_mlp_head(cls_in_channel, cls_hidden, num_classes)
        else:
            if self.pre_extract_path is not None:
                file_name = f"scannet_607_{'clip-ViT-B16' if self.language_type == 'clip' else 'bert-base-uncased'}_id.pth"
                self.register_buffer("text_embeds", torch.load(Path(self.pre_extract_path) / file_name).float())
            else:
                self.int2cat = json.load(open(cfg.model.vocab_path, "r"))
                if self.language_type == "clip":
                    self.clip_head = clip.load("ViT-B/16")
                    self.text_embeds = self.clip_head.encode_text(clip.tokenize(self.int2cat)).detach()
                elif self.language_type == "bert":
                    self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
                    self.bert_config = BertConfig(
                        hidden_size=768, num_hidden_layers=3, num_attention=12, type_vocab_size=2
                    )
                    self.model = BertModel.from_pretrained("bert-base-uncased", config=self.bert_config)
                    self.encoded_input = self.tokenizer(
                        self.int2cat, padding=True, truncation=True, add_special_tokens=True, return_tensors="pt"
                    )
                    self.text_embeds = self.model(**self.encoded_input).last_hidden_state
                    self.text_embeds = self.text_embeds.detach()
                else:
                    raise NotImplementedError
        self.dropout = nn.Dropout(0.1)

    def forward(self, data_dict):
        # prepare dict
        if 'cur_step' not in data_dict.keys():
            data_dict['cur_step'] = 1
            data_dict['total_steps'] = 1

        obj_pcds = data_dict["obj_fts"]
        batch_size, num_objs, _, _ = obj_pcds.size()
        if self.model_name == "pointnext":
            obj_locs = rearrange(obj_pcds[..., :3], 'b o p d -> (b o) p d')
            obj_fts = rearrange(obj_pcds[..., 3:], 'b o p d -> (b o) d p').contiguous()
            obj_embeds = self.point_feature_extractor(obj_locs, obj_fts, type="cls")
        elif self.model_name == "pointnet++":
            obj_pcds = rearrange(obj_pcds, 'b o p d -> (b o) p d')
            obj_embeds = self.point_feature_extractor(obj_pcds)
        elif self.model_name == "pointmlp":
            obj_pcds = rearrange(obj_pcds, 'b o p d -> (b o) p d')
            obj_embeds = self.point_feature_extractor(obj_pcds)
        obj_embeds = self.dropout(obj_embeds)
        if self.cfg.model.open_vocab:
            logits = obj_embeds @ self.text_embeds.t()
            data_dict["obj_logits"] = rearrange(logits, '(b o) c -> b o c', b=batch_size)
        else:
            data_dict["obj_logits"] = rearrange(self.cls_head(obj_embeds), '(b o) d -> b o d', b=batch_size)
        return data_dict

    def get_opt_params(self):
        optimizer_grouped_parameters = []
        optimizer_grouped_parameters.append({
            "params": self.parameters(),
            "weight_decay": self.cfg.solver.get("weight_decay", 0.0),
            "lr": self.cfg.solver.lr
        })
        return optimizer_grouped_parameters
