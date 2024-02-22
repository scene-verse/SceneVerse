import os
import glob
import importlib
import functools
import torch
from typing import Any
from accelerate.logging import get_logger
from accelerate.state import PartialState
from accelerate.utils import recursively_apply
from accelerate.utils.constants import TORCH_DISTRIBUTED_OPERATION_TYPES
from accelerate.utils.dataclasses import DistributedType

logger = get_logger(__name__)


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

# using wonder's beautiful simplification: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


# def import_all(exclude_list=None):
#     if exclude_list is None:
#         exclude_list = ["__init__.py", "build.py"]
#     print(f"file: {__file__}")
#     current_directory = os.path.dirname(__file__)
#     module_names = [
#         os.path.splitext(file)[0] for file in os.listdir(current_directory)
#         if file.endswith(".py") and file not in exclude_list
#     ]
#     for module_name in module_names:
#         module = importlib.import_module(f".{module_name}", package=__name__)
#         globals().update({name: getattr(module, name) for name in getattr(module, '__all__', [])})
#     __all__ = [name for name in globals() if not name.startswith("_")]


def _gpu_gather_object(object: Any):
    # by JY Huang: re-implement the method for gathering non-tensor objects
    output_objects = [None for _ in range(PartialState().num_processes)]
    torch.distributed.all_gather_object(output_objects, object)
    if isinstance(object, (list, tuple)):
        output_list = []
        for item in output_objects:
            output_list.extend(item)
        return output_list
    elif isinstance(object, dict):
        template = output_objects[0]
        output_dict = {}
        for k, v in template.items():
            output_dict[k] = []
            for item in output_objects:
                if isinstance(item[k], list):
                    output_dict[k].extend(item[k])
                else:
                    output_dict[k].append(item[k])
        return output_dict


def gather_object(object: Any):
    """
    Recursively gather object in a nested list/tuple/dictionary of objects from all devices.

    Args:
        object (nested list/tuple/dictionary of picklable object):
            The data to gather.

    Returns:
        The same data structure as `object` with all the objects sent to every device.
    """
    if PartialState().distributed_type == DistributedType.TPU:
        raise NotImplementedError("gather objects in TPU is not supported")
    elif PartialState().distributed_type in TORCH_DISTRIBUTED_OPERATION_TYPES:
        return _gpu_gather_object(object)
    else:
        return object


def gather_for_metrics(accelerator, input_data):
    """
    by JY Huang: re-implement this method for gathering non-tensor objects
    Refer source code to https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.gather_for_metrics
    """

    try:
        recursively_apply(lambda x: x, input_data, error_on_other_type=True)
        all_tensors = True
    except TypeError:
        all_tensors = False

    if not all_tensors:
        data = gather_object(input_data)
    else:
        data = accelerator.gather(input_data)

    try:
        if accelerator.gradient_state.end_of_dataloader:
            # at the end of a dataloader, `gather_for_metrics` regresses to
            # `gather` unless the dataset has a remainder so log.
            if accelerator.gradient_state.remainder == -1:
                logger.info(
                    "The used dataset had no length, returning gathered tensors. You should drop the remainder yourself."
                )
                return data
            elif accelerator.gradient_state.remainder > 0:
                # Last batch needs to be truncated on distributed systems as it contains additional samples
                def _adjust_samples(tensor):
                    return tensor[: accelerator.gradient_state.remainder] if tensor is not None else None
                if all_tensors:
                    # This only applies to tensors, as defined in `recursively_apply`
                    return recursively_apply(_adjust_samples, data)
                else:
                    if isinstance(data, (list, tuple)):
                        return _adjust_samples(data)
                    elif isinstance(data, dict):
                        return {k: _adjust_samples(v) for k, v in data.items()}
                    else:
                        raise NotImplementedError(f"Non-tensor gather only supports list, tuple or dict")
            else:  # remainder is 0
                # no remainder even though at end of dataloader, so nothing to do.
                return data
        else:
            # Not at the end of the dataloader, no need to adjust the tensors
            return data
    except Exception:
        # Dataset had no length or raised an error
        return data
    
def gather_dict(accelerator, data_dict):
    data_dict_non_tensor = {k : v for k, v in data_dict.items() if not isinstance(v, torch.Tensor)}
    data_dict_non_tensor = gather_for_metrics(accelerator, data_dict_non_tensor)
    data_dict = {k : v for k, v in data_dict.items() if isinstance(v, torch.Tensor)}
    data_dict = gather_for_metrics(accelerator, data_dict)
    data_dict.update(data_dict_non_tensor)
    return data_dict
