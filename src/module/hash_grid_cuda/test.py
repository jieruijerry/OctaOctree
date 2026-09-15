import os
import time
import torch
from torch.utils.cpp_extension import load

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 设置为1以便于调试CUDA错误
# os.environ["TORCH_USE_CUDA_DSA"] = "1"  # 启用CUDA DSA（Device Side Allocation）


N, L, D = 2**20, 4, 8
M = 19
base_resolution = 32
per_level_scale = 2.0

# 自动编译 & 加载 Extension
hash_grid = load(
    name="hash_grid_cuda",
    sources=["hash_grid_bindings.cpp", "hash_grid_cuda.cu"],
    extra_cflags=[
        '-O3',
        "-DLEVELS={}".format(L),
        "-DDIMENSIONS={}".format(D),
        "-DLOG_HASHMAP_SIZE={}".format(M),
        "-DBASE_RESOLUTION={}".format(base_resolution),
        "-DPER_LEVEL_SCALE={}".format(per_level_scale),
        "-DLAYER_REDUCE={}".format("CONCAT"),
    ],
    extra_cuda_cflags=[
        "-O3", "-g", "-lineinfo", "-Xcompiler", "-rdynamic",
        "-DLEVELS={}".format(L),
        "-DDIMENSIONS={}".format(D),
        "-DLOG_HASHMAP_SIZE={}".format(M),
        "-DBASE_RESOLUTION={}".format(base_resolution),
        "-DPER_LEVEL_SCALE={}".format(per_level_scale),
        "-DLAYER_REDUCE={}".format("CONCAT"),
        "-DTHREADS=128",
    ],
    verbose=True,  # 可看到详细编译过程
)

# Create random input
grids = []
for l in range(L):
    res = int(base_resolution * (per_level_scale ** l))
    grid_size = (res + 1) ** 3
    if grid_size > 2 ** M:
        grid = torch.randn(2 ** M, 1, D, device='cuda', dtype=torch.float32)
    else:
        grid = torch.randn(grid_size, 1, D, device='cuda', dtype=torch.float32)
    grids.append(grid)

pos = torch.rand(N, 3, device='cuda', dtype=torch.float32)

torch.cuda.synchronize()
t0 = time.time()

enc = hash_grid.forward(pos, grids)

torch.cuda.synchronize()
t1 = time.time()

print(f"CUDA Hash Grid time: {t1 - t0:.4f} seconds")