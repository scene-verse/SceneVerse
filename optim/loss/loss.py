import torch.nn as nn
import torch.nn.functional as F
from fvcore.common.registry import Registry


LOSS_REGISTRY = Registry("loss")

def og3d_loss(data_dict):
    return F.cross_entropy(data_dict["og3d_logits"], data_dict["tgt_object_id"].squeeze(1))


def og3d_multi_loss(data_dict):
    return F.binary_cross_entropy_with_logits(
        data_dict["og3d_logits"],
        data_dict["tgt_object_id"].float(),
        reduction="sum") / float(data_dict["tgt_object_id"].shape[0])


def txt_cls_multi_loss(data_dict):
    return F.binary_cross_entropy_with_logits(
        data_dict["txt_cls_logits"],
        data_dict["tgt_object_label"].float(),
        reduction='sum') / float(data_dict["tgt_object_label"].shape[0])


def obj_cls_raw_loss(data_dict):
    return (
        F.cross_entropy(
            data_dict["obj_cls_raw_logits"].permute(0, 2, 1), data_dict["obj_labels"], reduction='none'
        ) * data_dict["obj_masks"]
    ).sum() / data_dict["obj_masks"].sum()


def obj_cls_pre_loss(data_dict):
    return (
        F.cross_entropy(
            data_dict["obj_cls_pre_logits"].permute(0, 2, 1), data_dict["obj_labels"], reduction='none'
        ) * data_dict["obj_masks"]
    ).sum() / data_dict["obj_masks"].sum()


def obj_cls_post_loss(data_dict):
    return (
        F.cross_entropy(
            data_dict["obj_cls_post_logits"].permute(0, 2, 1), data_dict["obj_labels"], reduction='none'
        ) * data_dict["obj_masks"]
    ).sum() / data_dict["obj_masks"].sum()


def answer_loss(data_dict):
    return F.binary_cross_entropy_with_logits(
            data_dict["answer_scores"], data_dict["answer_label"].float(), reduction='sum'
        ) / data_dict["answer_scores"].shape[0]


def lm_cls_loss(data_dict):
    target_labels = data_dict["masked_lm_labels"]
    target_labels = target_labels.view(-1, target_labels.size(-1)) if len(target_labels.size()) == 3 else target_labels
    return F.cross_entropy(
            data_dict["txt_lm_cls_logits"].permute(0, 2, 1), target_labels, ignore_index=-1
        )


def obj_cls_pre_loss_mask(data_dict):
    return (
        F.cross_entropy(
            data_dict["obj_cls_pre_logits"].permute(0, 2, 1), data_dict["obj_labels"], reduction='none'
        ) * data_dict["obj_masks"] * data_dict["obj_sem_masks"].logical_not()
    ).sum() / (data_dict["obj_masks"] * data_dict["obj_sem_masks"].logical_not()).sum()


def obj_cls_pre_loss_unmask(data_dict):
    return (
        F.cross_entropy(
            data_dict["obj_cls_pre_logits"].permute(0, 2, 1), data_dict["obj_labels"], reduction='none'
        ) * data_dict["obj_masks"] * data_dict["obj_sem_masks"]
    ).sum() / (data_dict["obj_masks"] * data_dict["obj_sem_masks"]).sum()


def obj_cls_post_loss_mask(data_dict):
    return (
        F.cross_entropy(
            data_dict["obj_cls_post_logits"].permute(0, 2, 1), data_dict["obj_labels"], reduction='none'
        ) * data_dict["obj_masks"] * data_dict["obj_sem_masks"].logical_not()
    ).sum() / (data_dict["obj_masks"] * data_dict["obj_sem_masks"].logical_not()).sum()


def obj_cls_post_loss_unmask(data_dict):
    return (
        F.cross_entropy(
            data_dict["obj_cls_post_logits"].permute(0, 2, 1), data_dict["obj_labels"], reduction='none'
        ) * data_dict["obj_masks"] * data_dict["obj_sem_masks"]
    ).sum() / (data_dict["obj_masks"] * data_dict["obj_sem_masks"]).sum()


def obj_cls_loss(data_dict, smoothing=0.3):
    return (
        F.cross_entropy(
            data_dict["obj_logits"].permute(0, 2, 1), data_dict["obj_labels"],
            reduction='none', label_smoothing=smoothing
        ) * data_dict["obj_masks"]
    ).sum() / data_dict["obj_masks"].sum()


def mse_loss(data_dict):
    return (
        ((data_dict["pred_images"] - data_dict["target_images"]) ** 2).mean()
    )


class Loss(nn.Module):
    def __init__(self, cfg):
        # e.g.  refer_loss_v1: ["og3d_loss", "txt_cls_loss", "obj_cls_raw_loss", "obj_cls_pre_loss", "obj_cls_post_loss"]
        #       qa_loss_v1: ["og3d_loss", "txt_cls_loss", "obj_cls_raw_loss", "obj_cls_pre_loss", "obj_cls_post_loss", "answer_loss"]
        #       pretrain_loss_v1: ["lm_cls_loss", "obj_cls_raw_loss", "obj_cls_pre_loss", "obj_cls_post_loss", "obj_cls_pre_loss_mask",
        #                           "obj_cls_pre_loss_unmask", "obj_cls_post_loss_mask", "obj_cls_post_loss_unmask"]
        super().__init__()
        self.all_keys = list(set(cfg.model.vis_loss_list + cfg.model.loss_list))
        self.selected_keys = cfg.model.loss_list

        self.loss_fn = {}
        for k in self.all_keys:
            if k in globals().keys():
                self.loss_fn[k] = globals()[k]
                print(f"Using {k} from loss.globals()")
            else:
                self.loss_fn[k] = LOSS_REGISTRY.get(k)(cfg)
                setattr(self, k, self.loss_fn[k]) # register the loss module, otherwise its parameters will not be the same device as the model
                print(f"Using {k} from Registry {LOSS_REGISTRY._name}")

    def forward(self, data_dict):
        all_losses = {}
        for k, fn in self.loss_fn.items():
            if k == 'txt_cls_loss' and 'txt_cls_label' not in data_dict: # compatible with old version of txt_cls_loss
                data_dict['txt_cls_label'] = data_dict["tgt_object_label"].squeeze(1)
            cur_loss = fn(data_dict)
            if not isinstance(cur_loss, list):
                cur_dict_loss = {k : cur_loss}
            else:
                cur_dict_loss = {k: cur_loss[0]}
                for ck, cv in cur_loss[1].items():
                    cur_dict_loss[k + "_" + ck] = cv
            all_losses.update(cur_dict_loss)
        
        selected_losses = {k: all_losses[k] for k in self.selected_keys}
        total_loss = sum(selected_losses.values())
        all_losses["total_loss"] = total_loss
        return total_loss, all_losses
