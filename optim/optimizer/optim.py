import torch.optim as optim

from fvcore.common.registry import Registry
OPTIM_REGISTRY = Registry("loss")

from common.type_utils import cfg2dict


def get_optimizer(cfg, params):
  if getattr(optim, cfg.solver.optim.name, None) is not None:
    optimizer = getattr(optim, cfg.solver.optim.name)(params, **cfg2dict(cfg.solver.optim.args))
  else:
    optimizer = OPTIM_REGISTRY.get(cfg.solver.optim.name)(params, **cfg2dict(cfg.solver.optim.args))
  return optimizer
