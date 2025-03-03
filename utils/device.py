import torch
import numpy as np

def get_device(force_cpu=False):
    if force_cpu:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
        return "mps"
    else:
        return "cpu"
    
def get_torch_and_np_dtypes(device, use_bfloat16=False):
    if device == "cuda":
        torch_dtype = torch.bfloat16 if use_bfloat16 else torch.float16
        np_dtype = np.float16
    elif device == "mps":
        torch_dtype = torch.bfloat16 if use_bfloat16 else torch.float16
        np_dtype = np.float16
    else:
        torch_dtype = torch.float32
        np_dtype = np.float32
    return torch_dtype, np_dtype
