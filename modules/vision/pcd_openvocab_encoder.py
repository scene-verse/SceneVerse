import os
import glob

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.build import VISION_REGISTRY
from modules.layers.pointnet import PointNetPP
from modules.layers.transformers import TransformerSpatialEncoderLayer
from modules.utils import get_mlp_head, layer_repeat, calc_pairwise_locs, get_mixup_function
from modules.weights import _init_weights_bert


@VISION_REGISTRY.register()
class PointOpenVocabEncoder(nn.Module):
    def __init__(self, cfg, backbone='pointnet++', hidden_size=768, path=None, freeze=False, dim_feedforward=2048,
                 num_attention_heads=12, spatial_dim=5, num_layers=4, dim_loc=6, pairwise_rel_type='center',
                 use_matmul_label=False, mixup_strategy=None, mixup_stage1=None, mixup_stage2=None,
                 lang_type='bert', lang_path=None, attn_type='spatial'):
        super().__init__()
        assert backbone in ['pointnet++']

        # build backbone
        if backbone == 'pointnet++':
            self.point_feature_extractor = PointNetPP(
                sa_n_points=[32, 16, None],
                sa_n_samples=[32, 32, None],
                sa_radii=[0.2, 0.4, None],
                sa_mlps=[[3, 64, 64, 128], [128, 128, 128, 256], [256, 256, 512, 768]],
            )
        elif backbone == 'pointnext':
            self.point_feature_extractor = PointNextEncoder(
                blocks=[1, 1, 1, 1, 1, 1],
                strides=[1, 2, 2, 2, 2, 1],
                sa_layers=2,
                sa_use_res=True,
                width=32,
                radius=0.15,
                radius_scaling=1.5,
                mlp_head=[1024, 768] if lang_type == 'bert' else []
            )

        # Open vocab grounding head
        vocab_file_name = f"scannet_607_{'bert-base-uncased' if lang_type == 'bert' else 'clip-ViT-B16'}_id.pth"
        self.register_buffer("text_features", torch.load(os.path.join(lang_path, vocab_file_name)))
        self.point_cls_head = lambda x: x @ self.text_features.t()
        self.dropout = nn.Dropout(0.1)

        self.attn_type = attn_type

        # freeze feature
        self.freeze = freeze
        if freeze:
            for p in self.parameters():
                p.requires_grad = False

        # build semantic cls embeds
        self.sem_cls_embed_layer = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                                 nn.LayerNorm(hidden_size),
                                                 nn.Dropout(0.1))

        # self.int2cat = json.load(
        #     open(os.path.join(glove_path, "annotations/meta_data/scannetv2_raw_categories.json"), 'r'))
        # self.cat2int = {w: i for i, w in enumerate(self.int2cat)}
        # self.cat2vec = json.load(open(os.path.join(glove_path, "annotations/meta_data/cat2glove42b.json"), 'r'))
        # self.register_buffer("int2mat", torch.ones(607, 300))
        # for i in range(607):
        #     self.int2mat[i, :] = torch.Tensor(self.cat2vec[self.int2cat[i]])

        self.use_matmul_label = use_matmul_label
        # build mask embedes
        self.sem_mask_embeddings = nn.Embedding(1, 768)

        # build spatial encoder layer
        if self.attn_type == 'spatial':
            pc_encoder_layer = TransformerSpatialEncoderLayer(hidden_size, num_attention_heads,
                                                              dim_feedforward=dim_feedforward,
                                                              dropout=0.1, activation='gelu',
                                                              spatial_dim=spatial_dim, spatial_multihead=True,
                                                              spatial_attn_fusion='cond')
            self.spatial_encoder = layer_repeat(pc_encoder_layer, num_layers)
            loc_layer = nn.Sequential(
                nn.Linear(dim_loc, hidden_size),
                nn.LayerNorm(hidden_size),
            )
            self.loc_layers = layer_repeat(loc_layer, 1)
            self.pairwise_rel_type = pairwise_rel_type
            self.spatial_dim = spatial_dim
        else:
            pass

        # # build mixup strategy
        # self.mixup_strategy = mixup_strategy
        # self.mixup_function = get_mixup_function(mixup_strategy, mixup_stage1, mixup_stage2)

        # load weights
        self.apply(_init_weights_bert)
        if path is not None:
            # pre_dict = {}
            # for name, p in self.named_parameters():
            #     pre_dict[name] = p
            # TODO: change this to accelerator loading multiple model files
            print("loaded")
            ckpts = glob.glob(os.path.join(path, '*.bin'))
            if len(ckpts) != 0:
                for ckpt in ckpts:
                    state_dict = torch.load(ckpt, map_location='cpu')
                    self.load_state_dict(state_dict, strict=False)
                print("loaded checkpoint files")
            elif path.endswith('.pth'):
                print("loaded checkpoint file")
                state_dict = torch.load(path)
                self.load_state_dict(state_dict, strict=False)
            # for name, p in self.named_parameters():
            #     if name in state_dict.keys():
            #         print(name, pre_dict[name] - layer_repeat(p, 1))
            # exit()

    def freeze_bn(self, m):
        for layer in m.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, obj_pcds, obj_locs, obj_masks, obj_sem_masks,
                obj_labels=None, cur_step=None, max_steps=None, **kwargs):
        if self.freeze:
            self.freeze_bn(self.point_feature_extractor)

        # get obj_embdes
        batch_size, num_objs, _, _ = obj_pcds.size()
        obj_embeds = self.point_feature_extractor(einops.rearrange(obj_pcds, 'b o p d -> (b o) p d'))
        # obj_sem_embeds = self.sem_cls_embed_layer(obj_embeds)
        # obj_sem_embeds = einops.rearrange(obj_sem_embeds, '(b o) d -> b o d', b=batch_size)
        obj_embeds = einops.rearrange(obj_embeds, '(b o) d -> b o d', b=batch_size)
        obj_embeds = self.dropout(obj_embeds)
        if self.freeze:
            obj_embeds = obj_embeds.detach()

        # get semantic cls embeds
        obj_sem_cls = F.softmax(self.point_cls_head(obj_embeds), dim=2).detach()

        # TODO: check if this sem_cls is still needed, switch this to cross attention
        # if self.mixup_strategy != None:
        #     obj_sem_cls_mix = self.mixup_function(obj_sem_cls, obj_labels, cur_step, max_steps)
        # else:
        #     obj_sem_cls_mix = obj_sem_cls
        # if self.use_matmul_label:
        #     obj_sem_cls_embeds = torch.matmul(obj_sem_cls_mix, self.int2mat)  # N, O, 607 matmul ,607, 300
        # else:
        #     obj_sem_cls_mix = torch.argmax(obj_sem_cls_mix, dim=2)
        #     obj_sem_cls_embeds = torch.Tensor(
        #         [self.cat2vec[self.int2cat[int(i)]] for i in obj_sem_cls_mix.view(batch_size * num_objs)])
        #     obj_sem_cls_embeds = obj_sem_cls_embeds.view(batch_size, num_objs, 300).cuda()
        # obj_sem_cls_embeds = self.sem_cls_embed_layer(obj_sem_cls_embeds)
        # obj_embeds = obj_embeds + obj_sem_embeds

        # # get semantic mask embeds
        # obj_embeds = obj_embeds.masked_fill(obj_sem_masks.unsqueeze(2).logical_not(), 0.0)
        # obj_sem_mask_embeds = self.sem_mask_embeddings(
        #     torch.zeros((batch_size, num_objs)).long().cuda()
        # ) * obj_sem_masks.logical_not().unsqueeze(2)
        # obj_embeds = obj_embeds + obj_sem_mask_embeds

        # record pre embedes
        # note: in our implementation, there are three types of embds, raw embeds from PointNet,
        # pre embeds after tokenization, post embeds after transformers
        obj_embeds_pre = obj_embeds

        # spatial reasoning, spatial attention transformer
        if self.attn_type == 'spatial':
            pairwise_locs = calc_pairwise_locs(obj_locs[:, :, :3], obj_locs[:, :, 3:],
                                               pairwise_rel_type=self.pairwise_rel_type, spatial_dist_norm=True,
                                               spatial_dim=self.spatial_dim)
            for i, pc_layer in enumerate(self.spatial_encoder):
                query_pos = self.loc_layers[0](obj_locs)
                obj_embeds = obj_embeds + query_pos
                obj_embeds, self_attn_matrices = pc_layer(obj_embeds, pairwise_locs,
                                                          tgt_key_padding_mask=obj_masks.logical_not())
        else:
            pass

        return obj_embeds, obj_embeds_pre, obj_sem_cls