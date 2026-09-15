import gc
import torch


def cuda_mem(tag="", device=None):
    if device is None:
        device = torch.cuda.current_device()
    torch.cuda.synchronize(device)

    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    peak = torch.cuda.max_memory_allocated(device) / 1024**3

    print(f"[{tag}] allocated={allocated:.3f}GB reserved={reserved:.3f}GB peak={peak:.3f}GB")


def tensor_mb(t: torch.Tensor) -> float:
    return t.numel() * t.element_size() / 1024**2


def list_live_cuda_tensors(min_mb=1.0, max_items=100):
    rows = []

    for obj in gc.get_objects():
        try:
            if not torch.is_tensor(obj):
                continue
            if not obj.is_cuda:
                continue

            mb = tensor_mb(obj)
            if mb < min_mb:
                continue

            rows.append((
                mb,
                tuple(obj.shape),
                str(obj.dtype),
                str(obj.device),
                obj.requires_grad,
                obj.is_leaf,
                type(obj.grad_fn).__name__ if obj.grad_fn is not None else None,
            ))
        except Exception:
            continue

    rows.sort(reverse=True, key=lambda x: x[0])

    print(f"Live CUDA tensors >= {min_mb} MB: {len(rows)}")
    for i, row in enumerate(rows[:max_items]):
        mb, shape, dtype, device, requires_grad, is_leaf, grad_fn = row
        print(
            f"{i:04d} | {mb:9.2f} MB | {device} | {dtype} | "
            f"shape={shape} | grad={requires_grad} | leaf={is_leaf} | grad_fn={grad_fn}"
        )