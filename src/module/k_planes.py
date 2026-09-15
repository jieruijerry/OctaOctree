import torch
import torch.nn as nn


class KPlanes(nn.Module):
    """
    K-Planes
    """
    index_offset = torch.tensor(
        [[
            [[0], [0]],
            [[1], [0]],
            [[0], [1]],
            [[1], [1]],
        ]],
        dtype=torch.int64
    )

    big_primes = torch.tensor(
        [[1], [19349663]],
        dtype=torch.int64
    )

    def __init__(self, config: dict) -> None:
        super(KPlanes, self).__init__()
        self.config = config

        # Move constants to device
        self.register_buffer("idx_offset", KPlanes.index_offset)
        self.register_buffer("primes", KPlanes.big_primes)

        self.n_dims = config["n_dims_to_encode"]
        self.v_dims = ((self.n_dims - 1) / 2) if not hasattr(self, "v_dims") else self.v_dims
        self.K = int(self.n_dims * self.v_dims)
        self.L = config["n_levels"]
        self.D = config["n_features_per_level"]

        # level -> [grid_index, side, feature_dim]
        self.grids = nn.ParameterList()
        self.weights = []
        self.resolutions = []
        self.grid_sizes = []

        # Calculate grid sizes
        for i in range(self.L):
            resolution = int(config["base_resolution"] * config["per_level_scale"] ** i)

            n_items = min((resolution + 1) ** 2, 1 << config["log2_hashmap_size"])
            
            grid = nn.Parameter(torch.zeros(
                (self.K, n_items, self.D),
                dtype=torch.float32,
            ))

            self.grids.append(grid)
            self.resolutions.append(resolution)
            self.grid_sizes.append(n_items)
    
    @property
    def output_dim(self) -> int:
        dim = self.D
        if self.config["level_reduce"] == "Concat":
            dim *= self.L
        if self.config["k_reduce"] == "Concat":
            dim *= self.K
        return dim
    
    def get_uv(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Calculate uv coordinates
        """
        assert pos.size(1) == self.n_dims, \
            f"KPlanes encounter {pos.size(1)}D input while expecting {self.n_dims}D"
        
        uv = []
        for i in range(self.n_dims):
            for j in range(i + 1, self.n_dims):
                uv.append(torch.cat([pos[:, i:i+1], pos[:, j:j+1]], dim=-1))
        uv = torch.stack(uv, dim=-1)
        return uv
        
    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Interpolate grid features onto si.
        """
        uv = self.get_uv(pos)

        features = []
        for i in range(self.L):
            resolution = self.resolutions[i]
            grid = self.grids[i]

            # Calculate base and offset
            # [N, 2, K]
            base = torch.floor(uv * resolution).to(torch.int64)
            # [N, 2, K]
            offset = uv * resolution - base
            comp_offset = 1 - offset

            # Calculate hash index
            # [N, 4, 2, K]
            index = base.unsqueeze(1) + self.idx_offset
            # [N, 4, K]
            index = self._index_func(index, i)
            # [4N, K]
            index = index.reshape(-1, self.K)

            # Calculate tri-lerp weight
            w0 = comp_offset[:, 0] * comp_offset[:, 1]
            w1 = offset[:, 0]      * comp_offset[:, 1]
            w2 = comp_offset[:, 0] * offset[:, 1]
            w3 = offset[:, 0]      * offset[:, 1]
            # [N, 4, K]
            weight = torch.stack([w0, w1, w2, w3], dim=1)

            assert (index >= 0).all() and (index < self.grid_sizes[i]).all(), "Index out of range in KPlanes access!"
            # Fetch grid features
            feature = []
            for i in range(self.K):
                # [4N, D]
                feature.append(grid[i][index[:, i]])
            # [N, 4, K, D]
            feature = torch.cat(feature, dim=-1).reshape(-1, 4, self.K, self.D)
            # [N, K, D]
            feature = (feature * weight.unsqueeze(-1)).sum(dim=1)

            if self.config["k_reduce"] == "Concat":
                feature = feature.reshape(-1, self.K * self.D)
            elif self.config["k_reduce"] == "Mean":
                feature = feature.mean(dim=1)
            elif self.config["k_reduce"] == "Product":
                feature = feature.prod(dim=1)
            else:
                raise NotImplementedError("Unknown k_reduce method: " + self.config["k_reduce"])

            features.append(feature)
        
        if self.config["level_reduce"] == "Concat":
            result = torch.cat(features, dim=-1)
        elif self.config["level_reduce"] == "Mean":
            result = torch.stack(features, dim=1).mean(dim=1)
        
        return result
    
    def _index_func(self, index: torch.Tensor, level: int) -> torch.Tensor:
        """
        Index function.
        """
        resolution = self.resolutions[level]
        if ((resolution + 1) ** 2) > (1 << self.config["log2_hashmap_size"]):
            result = (index * self.primes).sum(dim=-2)
        else:
            result = (resolution + 1) * index[:, :, 0] + index[:, :, 1]
        result = result % self.grid_sizes[level]
        
        return result


class PVPlanes(KPlanes):
    """
    XV-Planes used in DNR
    """
    def __init__(self, config: dict, v_dims: int) -> None:
        self.v_dims = v_dims
        super(PVPlanes, self).__init__(config)
        
    def get_uv(self, posv: torch.Tensor) -> torch.Tensor:
        """
        Calculate uv coordinates
        """
        assert posv.size(1) == (self.n_dims + self.v_dims), \
            f"PVPlanes encounter {posv.size(1)}D variable input while expecting {self.n_dims}D + {self.v_dims}D"
        
        pos = posv[:, :self.n_dims]
        v = posv[:, self.n_dims:]
        
        uv = []
        for i in range(self.n_dims):
            for j in range(self.v_dims):
                uv.append(torch.cat([pos[:, i:i+1], v[:, j:j+1]], dim=-1))
        uv = torch.stack(uv, dim=-1)
        return uv