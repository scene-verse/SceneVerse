import math
from torch.optim.lr_scheduler import LambdaLR


def warmup_cosine(step, warmup_step, total_step, minimum_ratio=1e-5):
    if step <= warmup_step and warmup_step > 0:
        return step / warmup_step
    return max(
        0.5 * (1 + math.cos((step - warmup_step) / (total_step - warmup_step) * math.pi)),
        minimum_ratio
    )


def warmup_exp(step, warmup_step, total_step, **kwargs):
    if step <= warmup_step and warmup_step > 0:
        return step / warmup_step
    return kwargs["gamma"] ** (step * 1. / (total_step - warmup_step))


def get_scheduler(cfg, optimizer, total_steps):
    warmup_steps = cfg.solver.sched.args.warmup_steps * cfg.num_gpu
    minimum_ratio = cfg.solver.sched.args.get("minimum_ratio", 1e-5)
    lambda_func = lambda step: globals()[cfg.solver.sched.name](
        step, warmup_steps, total_steps, minimum_ratio=minimum_ratio
    )
    return LambdaLR(optimizer=optimizer, lr_lambda=lambda_func)
