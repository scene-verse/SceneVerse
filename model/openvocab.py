import numpy as np
import torch
import torch.nn as nn
from einops import einsum

from model.build import MODEL_REGISTRY, BaseModel
from modules.build import build_module
from optim.utils import no_decay_param_group


@MODEL_REGISTRY.register()
class OpenVocab(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.lang_encoder = build_module("language", self.cfg.model.language)
        self.point_encoder = build_module("vision", self.cfg.model.vision)
        self.unified_encoder = build_module("grounding", self.cfg.model.grounding)
        self.head_list = self.cfg.model.heads.head_list
        for head in self.head_list:
            setattr(self, head, build_module("heads", getattr(self.cfg.model.heads, head)))
        self.use_scene_cap = self.cfg.data.args.get("use_scene_cap", False)
        if self.use_scene_cap:
            self.object_pool = lambda x : x.mean(dim=1)

    def forward(self, data_dict):
        # prepare dict
        if 'cur_step' not in data_dict.keys():
            data_dict['cur_step'] = 1
            data_dict['total_steps'] = 1
        # basic feature extractor
        # point_features_pre_spatial is point features before spatial reasonging

        lang_basic_features = self.lang_encoder(data_dict['txt_ids'], data_dict['txt_masks'])
        if self.use_scene_cap:
            scene_txt_ids = data_dict['scene_txt_ids']
            scene_txt_masks = data_dict['scene_txt_masks']
            scene_lang_basic_features = self.lang_encoder(scene_txt_ids, scene_txt_masks)
            data_dict['scene_text_embed'] = scene_lang_basic_features[:, 0]

        if not "Scene" in self.cfg.model.vision.name:
            point_basic_features, point_features_pre, obj_cls_raw_logits = self.point_encoder(data_dict['obj_fts'].float(),
                                                                                                data_dict['obj_locs'],
                                                                                                data_dict['obj_masks'],
                                                                                                data_dict['obj_sem_masks'],
                                                                                                data_dict['obj_labels'],
                                                                                                data_dict['cur_step'],
                                                                                                data_dict['total_steps'])
        else:
            point_basic_features, point_features_pre, obj_cls_raw_logits = self.point_encoder(data_dict)

        if self.use_scene_cap:
            scene_feature = self.object_pool(point_basic_features)
            data_dict["scene_embed"] = scene_feature

        if self.cfg.model.inter == "before":
            data_dict["inter_text_embed"] = lang_basic_features[:, 0]
            data_dict["inter_obj_embeds"] = point_basic_features

        # unifed language entity transformer
        language_fuse_feature, point_fuse_feature = self.unified_encoder(lang_basic_features, data_dict['txt_masks'],
                                                                         point_basic_features, data_dict['obj_locs'],
                                                                         data_dict['obj_masks'])
        if self.cfg.model.inter != "before":
            data_dict["inter_text_embed"] = language_fuse_feature[:, 0]
            data_dict["inter_obj_embeds"] = point_fuse_feature

        # # TODO: check if this is correct and if an additional mlp head is needed
        language_summarize_feature = language_fuse_feature[:, 0]
        data_dict["intra_text_embed"] = language_summarize_feature
        data_dict["intra_obj_embeds"] = point_fuse_feature

        data_dict['obj_cls_raw_logits'] = obj_cls_raw_logits
        data_dict['og3d_logits'] = einsum(point_fuse_feature, language_summarize_feature, "b o d, b d -> b o")

        # task head
        if getattr(self, "ground_head", None) is not None:
            txt_cls_logits, obj_cls_post_logits, obj_cls_pre_logits, og3d_logits = self.ground_head(language_fuse_feature,
                                                                                                point_fuse_feature,
                                                                                                point_features_pre,
                                                                                                data_dict['obj_masks'])
            data_dict['txt_cls_logits'] = txt_cls_logits
            data_dict['obj_cls_post_logits'] = obj_cls_post_logits
            data_dict['obj_cls_pre_logits'] = obj_cls_pre_logits
            # reload og3d_logits for head concatenated finetuning
            data_dict['og3d_logits'] = og3d_logits

        if getattr(self, "qa_head", None) is not None:
            answer_scores = self.qa_head(point_fuse_feature, data_dict['obj_masks'], language_fuse_feature,
                                         data_dict['txt_masks'])
            data_dict['answer_scores'] = answer_scores
        if getattr(self, "pretrain_head", None) is not None:
            output = self.pretrain_head(language_fuse_feature, point_fuse_feature)
            if isinstance(output, tuple):
                txt_lm_cls_logits, obj_lm_cls_logits = output
                data_dict['obj_cls_post_logits'] = obj_lm_cls_logits
            else:
                txt_lm_cls_logits = output
            data_dict['txt_lm_cls_logits'] = txt_lm_cls_logits

        return data_dict

    def get_opt_params(self):
        def get_lr(cfg, default_lr):
            return default_lr if cfg.get("lr") is None else cfg.get("lr")

        optimizer_grouped_parameters = []
        optimizer_grouped_parameters += no_decay_param_group(self.lang_encoder.named_parameters(),
                                                             get_lr(self.cfg.model.language, self.cfg.solver.lr))
        optimizer_grouped_parameters += no_decay_param_group(self.point_encoder.named_parameters(),
                                                             get_lr(self.cfg.model.vision, self.cfg.solver.lr))
        optimizer_grouped_parameters += no_decay_param_group(self.unified_encoder.named_parameters(),
                                                             get_lr(self.cfg.model.grounding, self.cfg.solver.lr))
        if "ground_head" in self.head_list:
            optimizer_grouped_parameters += no_decay_param_group(
                self.ground_head.named_parameters(), get_lr(self.cfg.model.heads.ground_head, self.cfg.solver.lr)
            )
        if "qa_head" in self.head_list:
            optimizer_grouped_parameters += no_decay_param_group(
                self.qa_head.named_parameters(), get_lr(self.cfg.model.heads.qa_head, self.cfg.solver.lr)
            )
        if "pretrain_head" in self.head_list:
            optimizer_grouped_parameters += no_decay_param_group(
                self.pretrain_head.named_parameters(), get_lr(self.cfg.model.heads.pretrain_head, self.cfg.solver.lr)
            )
        return optimizer_grouped_parameters


@MODEL_REGISTRY.register()
class OpenVocabPerScene(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.lang_encoder = build_module("language", self.cfg.model.language)
        self.point_encoder = build_module("vision", self.cfg.model.vision)
        self.unified_encoder = build_module("grounding", self.cfg.model.grounding)
        self.head_list = self.cfg.model.heads.head_list
        for head in self.head_list:
            setattr(self, head, build_module("heads", getattr(self.cfg.model.heads, head)))

    def forward(self, data_dict):
        # prepare dict
        if 'cur_step' not in data_dict.keys():
            data_dict['cur_step'] = 1
            data_dict['total_steps'] = 1

        use_per_scene = (len(data_dict['txt_ids'].shape) == 3)

        if use_per_scene:
            B, L, _ = data_dict['txt_ids'].shape
            B, O = data_dict['obj_masks'].shape

        # basic feature extracter
        # point_features_pre_spatial is point features before spatial reasonging
        txt_ids = data_dict['txt_ids'].view(B * L, -1) if use_per_scene else data_dict['txt_ids']
        txt_masks = data_dict['txt_masks'].view(B * L, -1) if use_per_scene else data_dict['txt_masks']

        lang_basic_features = self.lang_encoder(txt_ids, txt_masks)   # (B, L), D
        if not "Scene" in self.cfg.model.vision.name:
            point_basic_features, point_features_pre, obj_cls_raw_logits = self.point_encoder(data_dict['obj_fts'].float(),
                                                                                                data_dict['obj_locs'],
                                                                                                data_dict['obj_masks'],
                                                                                                data_dict['obj_sem_masks'],
                                                                                                data_dict['obj_labels'],
                                                                                                data_dict['cur_step'],
                                                                                                data_dict['total_steps'])
        else:
            point_basic_features, point_features_pre, obj_cls_raw_logits = self.point_encoder(data_dict)

        point_basic_features = point_basic_features.unsqueeze(1).repeat(1, L, 1, 1) \
                                        if use_per_scene else point_basic_features
        point_basic_features = point_basic_features.view(B * L, O, point_basic_features.shape[-1]) \
                                        if use_per_scene else point_basic_features
        if use_per_scene:
            obj_locs = data_dict['obj_locs'].unsqueeze(1).repeat(1, L, 1, 1)
            obj_locs = obj_locs.view(B * L, O, obj_locs.shape[-1])
            obj_masks = data_dict['obj_masks'].unsqueeze(1).repeat(1, L, 1)
            obj_masks = obj_masks.view(B * L, O)
        else:
            obj_locs = data_dict['obj_locs']
            obj_masks = data_dict['obj_masks']

        if self.cfg.model.inter == "before":
            data_dict["inter_text_embed"] = lang_basic_features[:, 0]
            data_dict["inter_obj_embeds"] = point_basic_features

        # unifed language entity transformer
        language_fuse_feature, point_fuse_feature = self.unified_encoder(lang_basic_features, txt_masks,
                                                                         point_basic_features, obj_locs,
                                                                         obj_masks)
        if self.cfg.model.inter != "before":
            data_dict["inter_text_embed"] = language_fuse_feature[:, 0]
            data_dict["inter_obj_embeds"] = point_fuse_feature

        # # TODO: check if this is correct and if an additional mlp head is needed
        language_summarize_feature = language_fuse_feature[:, 0]
        data_dict["intra_text_embed"] = language_summarize_feature
        data_dict["intra_obj_embeds"] = point_fuse_feature

        data_dict['obj_cls_raw_logits'] = obj_cls_raw_logits
        data_dict['og3d_logits'] = einsum(point_fuse_feature, language_summarize_feature, "b o d, b d -> b o")

        if use_per_scene:
            data_dict['og3d_logits'] = data_dict['og3d_logits'].view(B, L, O)

        # # task head
        # if getattr(self, "ground_head", None) is not None:
        #     txt_cls_logits, obj_cls_post_logits, obj_cls_pre_logits, og3d_logits = self.ground_head(language_fuse_feature,
        #                                                                                         point_fuse_feature,
        #                                                                                         point_features_pre,
        #                                                                                         data_dict['obj_masks'])
        #     data_dict['txt_cls_logits'] = txt_cls_logits
        #     data_dict['obj_cls_post_logits'] = obj_cls_post_logits
        #     data_dict['obj_cls_pre_logits'] = obj_cls_pre_logits
        #     # reload og3d_logits for head concatenated finetuning
        #     data_dict['og3d_logits'] = og3d_logits
        #
        if getattr(self, "qa_head", None) is not None:
            answer_scores = self.qa_head(point_fuse_feature, data_dict['obj_masks'], language_fuse_feature,
                                         data_dict['txt_masks'])
            data_dict['answer_scores'] = answer_scores
        if getattr(self, "pretrain_head", None) is not None:
            output = self.pretrain_head(language_fuse_feature, point_fuse_feature)
            if isinstance(output, tuple):
                txt_lm_cls_logits, obj_lm_cls_logits = output
                data_dict['obj_cls_post_logits'] = obj_lm_cls_logits
            else:
                txt_lm_cls_logits = output
            data_dict['txt_lm_cls_logits'] = txt_lm_cls_logits
        return data_dict

    def get_opt_params(self):
        def get_lr(cfg, default_lr):
            return default_lr if cfg.get("lr") is None else cfg.get("lr")

        optimizer_grouped_parameters = []
        optimizer_grouped_parameters += no_decay_param_group(self.lang_encoder.named_parameters(),
                                                             get_lr(self.cfg.model.language, self.cfg.solver.lr))
        optimizer_grouped_parameters += no_decay_param_group(self.point_encoder.named_parameters(),
                                                             get_lr(self.cfg.model.vision, self.cfg.solver.lr))
        optimizer_grouped_parameters += no_decay_param_group(self.unified_encoder.named_parameters(),
                                                             get_lr(self.cfg.model.grounding, self.cfg.solver.lr))
        if "ground_head" in self.head_list:
            optimizer_grouped_parameters += no_decay_param_group(
                self.ground_head.named_parameters(), get_lr(self.cfg.model.heads.ground_head, self.cfg.solver.lr)
            )
        if "qa_head" in self.head_list:
            optimizer_grouped_parameters += no_decay_param_group(
                self.qa_head.named_parameters(), get_lr(self.cfg.model.heads.qa_head, self.cfg.solver.lr)
            )
        if "pretrain_head" in self.head_list:
            optimizer_grouped_parameters += no_decay_param_group(
                self.pretrain_head.named_parameters(), get_lr(self.cfg.model.heads.pretrain_head, self.cfg.solver.lr)
            )
        return optimizer_grouped_parameters