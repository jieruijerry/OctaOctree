import torch
import torch.nn as nn


def octa_dir_to_uv(dir: torch.Tensor):
    """
    Convert ray directions to octahedral uv [-1,1]^2
    """
    xx, yy, zz = dir[..., 0:1], dir[..., 1:2], dir[..., 2:3]
    dir = dir / (torch.abs(xx) + torch.abs(yy) + torch.abs(zz) + 1e-8)

    xx, yy, zz = dir[..., 0], dir[..., 1], dir[..., 2]
    u = torch.where(zz > 0, (1 - torch.abs(yy)) * torch.sign(xx), xx)
    v = torch.where(zz > 0, (1 - torch.abs(xx)) * torch.sign(yy), yy)

    uv = torch.stack([u, v], dim=-1)

    return torch.clamp(uv, -1 + 1e-6, 1 - 1e-6)

def octa_lerp(base: torch.Tensor, frac: torch.Tensor, resolution: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Octahedral interpolation.
    base: [N, 8, 2]
    frac: [N, 8, 2]
    return: index [N, 8, 3, 2], weight [N, 8, 3]
    """
    # Conditions [N, 8, 1]
    # Phase 2, 4: use slash diagonal
    slash = ((base[..., 0:1] < 0).logical_xor(base[..., 1:2] < 0))
    # u > v: use right corner
    ugtv = (frac[..., 0:1] > frac[..., 1:2])
    # u + v > 1: use top-right corner
    upvgt1 = (frac[..., 0:1] + frac[..., 1:2] > 1)

    # [N, 8, 1, 2]: Corners
    base = base[..., None, :]
    top = base.clone()
    top[..., 1] += 1
    right = base.clone()
    right[..., 0] += 1
    topright = base + 1

    # [N, 8, 3, 2]: Triangle vertices
    index = torch.where(
        slash[..., None],  # [N, 8, 1, 1]
        torch.where(ugtv[..., None], torch.cat([base, topright, right], dim=-2), torch.cat([base, topright, top], dim=-2)),
        torch.where(upvgt1[..., None], torch.cat([right, top, topright], dim=-2), torch.cat([right, top, base], dim=-2))
    )

    # Calculate weights
    weight = torch.where(
        slash,
        torch.where(ugtv,
            torch.cat([1 - frac[..., 0:1], frac[..., 1:2], frac[..., 0:1] - frac[..., 1:2]], dim=-1),
            torch.cat([1 - frac[..., 1:2], frac[..., 0:1], frac[..., 1:2] - frac[..., 0:1]], dim=-1)),
        torch.where(upvgt1,
            torch.cat([1 - frac[..., 1:2], 1 - frac[..., 0:1], frac[..., 0:1] + frac[..., 1:2] - 1], dim=-1),
            torch.cat([frac[..., 0:1], frac[..., 1:2], 1 - frac[..., 0:1] - frac[..., 1:2]], dim=-1))
    )

    # Flip edge vertex indices
    # [N, 8, 3, 1]: Whether the vertex is on the right/top/left/bottom edge
    left_right = index[..., 0:1].abs() == (resolution // 2)
    top_bottom = index[..., 1:2].abs() == (resolution // 2)
    slash = ((index[..., 0:1] < 0).logical_xor(index[..., 1:2] < 0))
    corner = left_right & top_bottom

    index = torch.where(
        corner, index.abs(),
        torch.where(
            slash & left_right,
            torch.cat([index[..., 0:1], -index[..., 1:2]], dim=-1),
            torch.where(
                slash & top_bottom,
                torch.cat([-index[..., 0:1], index[..., 1:2]], dim=-1),
                index
            )
        )
    )

    # Convert index to positive
    index = index + (resolution // 2)

    return index, weight


class OctaGrid(nn.Module):
    """
    Multi-resolution octahedral grid encoding.
    - 1st / 3rd phase: top-left / bottom-right triangle (backslash)
    - 2nd / 4th phase: top-right / bottom-left triangle (slash)

    A-------B-------A   A( 0, 0, 1)
    |      /|\      |   B( 0, 1, 0)
    |    /  |  \    |   C(-1, 0, 0)
    |  /+---+    \  |   D( 0, 0,-1)
    |/  | / |      \|   E( 1, 0, 0)
    C---+---D-------E   F( 0,-1, 0)
    |\      |      /|
    |  \    |    /  |   ^ +Y/+V
    |    \  |  /    |   |
    |      \|/      |   o--> +X/+U
    A-------F-------A   (out) +Z
    """

    _primes = torch.tensor([
        1, 2654435761, 805459861,
        3674653429, 2097192037, 1434869437,
        2165219737, 122420729, 163227661,
        217636919, 290182597
    ], dtype=torch.int64)

    _spatial_offset = torch.tensor(
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

    def __init__(self, config: dict, bbox: torch.Tensor) -> None:
        super(OctaGrid, self).__init__()
        self.config = config
        assert config["n_dims_to_encode"] == 2, "OctaGrid only supports 2D directional encoding!"
        if config["spatial_resolution"] > 16:
            raise Warning("Spatial resolution greater than 16 may cause giant memory overhead, please check your config!")

        self.L = config["n_levels"]
        self.D = config["n_features_per_level"]
        self.spatial_res = config["spatial_resolution"]

        # level -> [grid_index, side, feature_dim]
        self.grids = nn.ParameterList()
        self.resolutions = []
        self.grid_sizes = []
        cardinals = torch.zeros((self.L, 3+2+1), dtype=torch.int64)

        # Calculate grid sizes
        for i in range(self.L):
            resolution = int(2 * config["per_level_scale"] ** i)
            cardinals[i, 0] = 1
            cardinals[i, 1] = cardinals[i, 0] * (self.spatial_res + 1)
            cardinals[i, 2] = cardinals[i, 1] * (self.spatial_res + 1)
            cardinals[i, 3] = cardinals[i, 2] * (self.spatial_res + 1)
            cardinals[i, 4] = cardinals[i, 3] * (resolution + 1)
            cardinals[i, -1] = cardinals[i, 4] * (resolution + 1)

            n_items = cardinals[i, -1]
            if n_items > (1 << config["log2_hashmap_size"]):
                n_items = 1 << config["log2_hashmap_size"]
            
            grid = nn.Parameter(torch.zeros(
                (n_items, self.D),
                dtype=torch.float32
            ))

            self.grids.append(grid)
            self.resolutions.append(resolution)
            self.grid_sizes.append(n_items)

        # Grid scale
        grid_scale = (bbox[1] - bbox[0]) / self.spatial_res

        # Move constants to device
        self.register_buffer("primes", OctaGrid._primes[:(3+2)])
        self.register_buffer("spatial_offset", OctaGrid._spatial_offset)
        self.register_buffer("cardinals", cardinals)
        self.register_buffer("grid_scale", grid_scale)

    @property
    def output_dim(self) -> int:
        if self.config["level_reduce"] == "Concat":
            return self.L * self.D
        return self.D
    
    def forward(self, pos: torch.Tensor, dir: torch.Tensor, wr_disp: torch.Tensor) -> torch.Tensor:
        """
        Interpolate grid features onto si.
        """
        features = []

        # Calculate spatial base and offset
        xi = torch.floor(pos * self.spatial_res).to(torch.int64)
        xf = pos * self.spatial_res - xi
        # Spatial index
        # [N, 8, 3]
        spatial_index = xi.unsqueeze(1) + self.spatial_offset

        # Calculate tri-lerp weight
        # [N, 8]
        spatial_weight = ((1.0 - self.spatial_offset) - xf.unsqueeze(1)).prod(dim=-1).abs()

        # Get direction perturbation
        # [N, 8, 3]
        xf = xf.unsqueeze(1) - self.spatial_offset
        perturb = xf * self.grid_scale

        # Add perturbation to direction
        # [N, 8, 3]
        dir_vec = dir[..., None, :] + perturb * wr_disp[..., None, :]
        dir = torch.where(
            (wr_disp.isinf() | wr_disp.isnan())[..., None, :],
            dir[..., None, :],
            dir_vec / torch.norm(dir_vec, dim=-1, keepdim=True)
        )

        # Calculate octahedral coordinate
        # [N, 8, 2]
        uv = octa_dir_to_uv(dir)

        for i in range(self.L):
            resolution = self.resolutions[i]
            grid = self.grids[i]

            # Calculate directional base and offset
            # [N, 8, 2]
            uvi = torch.floor(uv * (resolution // 2)).to(torch.int64)
            uvf = uv * (resolution // 2) - uvi

            # Directional index
            # [N, 8, 3, 2] / [N, 8, 3]
            dir_index, dir_weight = octa_lerp(uvi, uvf, resolution)

            # [N, 8, 3, 5] / [N, 8, 3, 1]
            index = torch.cat([
                spatial_index[..., :, None, :].repeat(1, 1, 3, 1),
                dir_index], dim=-1)
            weight = (spatial_weight[..., :, None] * dir_weight)[..., None]

            # [N, 8, 3]
            index = self._hash_func(index, i)

            assert (index >= 0).all() and (index < self.grid_sizes[i]).all(), "Index out of range in HashGrid access!"

            # Fetch grid features
            # [N, 8, 3, D]
            feature = grid[index.reshape(-1)].reshape(-1, 8, 3, self.D)
            # [N, D]
            feature = (feature * weight).sum(dim=(1, 2))

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
        cardinal = self.cardinals[level]

        if cardinal[-1] > (1 << self.config["log2_hashmap_size"]):
            ind = (index * self.primes) & 0xFFFFFFFF
            for i in range(1, 3+2):
                ind[..., 0] ^= ind[..., i]
            result = ind[..., 0] % (1 << self.config["log2_hashmap_size"])
        else:
            result = torch.sum(index * cardinal[:-1], dim=-1)
        
        return result
    