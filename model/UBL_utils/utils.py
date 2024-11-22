import torch 
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
from transformers.integrations.deepspeed import is_deepspeed_available
from transformers.utils import (
    ContextManagers,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_scipy_available,
    logging,
    replace_return_docstrings,
)
import numpy as np
import math

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def is_fp16_enabled():
    # Autocast world
    fp16_enabled = torch.get_autocast_gpu_dtype() == torch.float16
    fp16_enabled = fp16_enabled and torch.is_autocast_enabled()

    return fp16_enabled


def is_deepspeed_initialized():
    if is_deepspeed_available():
        return False
    else:
        try:
            import deepspeed

            # This is not available in all DeepSpeed versions.
            return deepspeed.utils.is_initialized()
        except Exception:
            return False
        

def collate_dense_tensors(samples: List[torch.Tensor], pad_v: float = 0) -> torch.Tensor:
    """
    Takes a list of tensors with the following dimensions:
        [(d_11, ..., d_1K),
         (d_21, ..., d_2K), ..., (d_N1, ..., d_NK)]
    and stack + pads them into a single tensor of:
    (N, max_i=1,N { d_i1 }, ..., max_i=1,N {diK})
    """
    if len(samples) == 0:
        return torch.Tensor()
    if len({x.dim() for x in samples}) != 1:
        raise RuntimeError(f"Samples has varying dimensions: {[x.dim() for x in samples]}")
    (device,) = tuple({x.device for x in samples})  # assumes all on same device
    max_shape = [max(lst) for lst in zip(*[x.shape for x in samples])]
    result = torch.empty(len(samples), *max_shape, dtype=samples[0].dtype, device=device)
    result.fill_(pad_v)
    for i in range(len(samples)):
        result_i = result[i]
        t = samples[i]
        result_i[tuple(slice(0, k) for k in t.shape)] = t
    return result


def flatten_final_dims(t: torch.Tensor, no_dims: int):
    return t.reshape(t.shape[:-no_dims] + (-1,))


def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def dict_multimap(fn, dicts):
    first = dicts[0]
    new_dict = {}
    for k, v in first.items():
        all_v = [d[k] for d in dicts]
        if isinstance(v, dict):
            new_dict[k] = dict_multimap(fn, all_v)
        else:
            new_dict[k] = fn(all_v)

    return new_dict

logger = logging.get_logger(__name__)
def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
    shape = weights.shape
    scale = scale / max(1, shape[1])

    if not is_scipy_available():
        logger.warning(
            "This init requires scipy, but scipy was not found, default to an approximation that might not be"
            " equivalent."
        )
        std = math.sqrt(scale)
        torch.nn.init.normal_(weights, std=std).clamp(min=0.0, max=2.0 * std)

    else:
        from scipy.stats import truncnorm

        std = math.sqrt(scale) / truncnorm.std(a=-2, b=2, loc=0, scale=1)
        samples = truncnorm.rvs(a=-2, b=2, loc=0, scale=std, size=weights.numel())
        samples = np.reshape(samples, shape)
        weights.copy_(torch.tensor(samples, device=weights.device))


def ipa_point_weights_init_(weights):
    with torch.no_grad():
        softplus_inverse_1 = 0.541324854612918
        weights.fill_(softplus_inverse_1)

