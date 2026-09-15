import torch
import drjit as dr
import gc

def empty_cache():
    torch.cuda.empty_cache()
    dr.flush_kernel_cache()
    dr.flush_malloc_cache()
    gc.collect()