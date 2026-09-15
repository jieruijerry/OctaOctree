import torch
import torch.nn as nn

import os
from torch.utils.cpp_extension import load


class HashGrid(nn.Module):
    """
    Multi-resolution Hash Grid
    """

    index_offset = torch.tensor(
        [[
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ]],
        dtype=torch.int64
    )

    big_primes = torch.tensor(
        [1, 19349663, 83492791],
        dtype=torch.int64
    )

    def __init__(self,
        config: dict,
        use_kernel: bool = False
    ) -> None:
        super(HashGrid, self).__init__()
        self.config = config
        self.use_kernel = use_kernel

        # Move constants to device
        self.register_buffer("idx_offset", HashGrid.index_offset)
        self.register_buffer("primes", HashGrid.big_primes)

        assert config["n_dims_to_encode"] == 3, "Hash grid only supports 3D"
        self.L = config["n_levels"]
        self.D = config["n_features_per_level"]

        # level -> [grid_index, side, feature_dim]
        self.grids = nn.ParameterList()
        self.resolutions = []
        self.grid_sizes = []

        # Calculate grid sizes
        for i in range(self.L):
            resolution = int(config["base_resolution"] * config["per_level_scale"] ** i)

            n_items = min((resolution + 1) ** 3, 1 << config["log2_hashmap_size"])
            
            grid = nn.Parameter(torch.zeros(
                (n_items, self.D),
                dtype=torch.float32
            ))

            self.grids.append(grid)
            self.resolutions.append(resolution)
            self.grid_sizes.append(n_items)
    
    @property
    def output_dim(self) -> int:
        if self.config["level_reduce"] == "Concat":
            return self.L * self.D
        return self.D
    
    def load_kernel(self, kernel_name: str = "hash_grid_cuda") -> None:
        """
        Load CUDA kernel for hash function.
        """
        self.use_kernel = True

        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.cuda_kernel = load(
            name=kernel_name,
            sources=[
                os.path.join(current_dir, "hash_grid_cuda", "hash_grid_bindings.cpp"),
                os.path.join(current_dir, "hash_grid_cuda", "hash_grid_cuda.cu")],
            extra_cflags=[
                '-O3',
                "-DLEVELS={}".format(self.L),
                "-DDIMENSIONS={}".format(self.config["n_features_per_level"]),
                "-DLOG_HASHMAP_SIZE={}".format(self.config["log2_hashmap_size"]),
                "-DBASE_RESOLUTION={}".format(self.config["base_resolution"]),
                "-DPER_LEVEL_SCALE={}".format(self.config["per_level_scale"]),
                "-DLAYER_REDUCE={}".format(self.config["level_reduce"].upper()),
            ],
            extra_cuda_cflags=[
                "-O3", "-g", "-lineinfo", "-Xcompiler", "-rdynamic",
                "-DLEVELS={}".format(self.L),
                "-DDIMENSIONS={}".format(self.config["n_features_per_level"]),
                "-DLOG_HASHMAP_SIZE={}".format(self.config["log2_hashmap_size"]),
                "-DBASE_RESOLUTION={}".format(self.config["base_resolution"]),
                "-DPER_LEVEL_SCALE={}".format(self.config["per_level_scale"]),
                "-DLAYER_REDUCE={}".format(self.config["level_reduce"].upper()),
            ],
            verbose=True,
        )
    
    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Interpolate grid features onto si.
        """
        features = []

        if self.use_kernel:
            return self.cuda_kernel.forward(pos.contiguous(), self.grids)

        for i in range(self.L):
            resolution = self.resolutions[i]
            grid = self.grids[i]

            # Calculate base and offset
            # [N, 3]
            base = torch.floor(pos * resolution).to(torch.int64)
            # [N, 3]
            offset = pos * resolution - base
            comp_offset = 1 - offset

            # Calculate hash index
            # [N, 8, 3]
            index = base.unsqueeze(1) + self.idx_offset
            # [N, 8]
            index = self._hash_func(index, i)
            # [8N]
            index = index.reshape(-1)

            # Calculate tri-lerp weight
            w0 = comp_offset[:, 0] * comp_offset[:, 1] * comp_offset[:, 2]
            w1 = offset[:, 0]      * comp_offset[:, 1] * comp_offset[:, 2]
            w2 = comp_offset[:, 0] * offset[:, 1]      * comp_offset[:, 2]
            w3 = offset[:, 0]      * offset[:, 1]      * comp_offset[:, 2]
            w4 = comp_offset[:, 0] * comp_offset[:, 1] * offset[:, 2]
            w5 = offset[:, 0]      * comp_offset[:, 1] * offset[:, 2]
            w6 = comp_offset[:, 0] * offset[:, 1]      * offset[:, 2]
            w7 = offset[:, 0]      * offset[:, 1]      * offset[:, 2]
            # [N, 8]
            weight = torch.stack([w0, w1, w2, w3, w4, w5, w6, w7], dim=1)

            assert (index >= 0).all() and (index < self.grid_sizes[i]).all(), "Index out of range in HashGrid access!"
            # Fetch grid features
            # [N, 8, D]
            feature = grid[index].reshape(-1, 8, self.D)
            # [N, D]
            feature = (feature * weight.unsqueeze(-1)).sum(dim=1)

            features.append(feature)
        
        if self.config["level_reduce"] == "Concat":
            result = torch.cat(features, dim=-1)
        elif self.config["level_reduce"] == "Mean":
            result = torch.stack(features, dim=-1).mean(dim=-1)
        else:
            raise NotImplementedError("Unknown level reduce method: " + self.config["level_reduce"])
        
        return result
    
    def _hash_func(self, index: torch.Tensor, level: int) -> torch.Tensor:
        """
        Hash function.
        """
        resolution = self.resolutions[level]

        if ((resolution + 1) ** 3) > (1 << self.config["log2_hashmap_size"]):
            result = (index * self.primes).sum(dim=-1) % \
                     (1 << self.config["log2_hashmap_size"])
        else:
            result = (resolution + 1) * (resolution + 1) * index[..., 0] + \
                     (resolution + 1) * index[..., 1] + \
                     index[..., 2]
        
        return result
    