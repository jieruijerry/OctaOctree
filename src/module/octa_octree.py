import torch
import torch.nn as nn
import numpy as np

import ocnn
from ocnn.octree import Octree, Points

import os
import time
from torch.utils.cpp_extension import load


def get_time():
    torch.cuda.synchronize()
    return time.time()


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


class _OctaOctreeQuery(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        cuda_kernel,
        pos,
        children,
        dir,
        disp,
        grid_scales,
        depths,
        resolutions,
        entries,
        hashmap_size,
        *features
    ):
        ctx.cuda_kernel = cuda_kernel
        ctx.children = children
        ctx.hashmap_size = hashmap_size
        ctx.save_for_backward(pos, dir, disp, grid_scales, depths, resolutions, entries, *features)

        return cuda_kernel.query_forward(
            pos,
            children,
            dir,
            disp,
            grid_scales,
            depths,
            resolutions,
            entries,
            hashmap_size,
            list(features)
        )

    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors
        pos, dir, disp, grid_scales, depths, resolutions, entries = saved[:7]
        features = list(saved[7:])

        grads = ctx.cuda_kernel.query_backward(
            grad_output.contiguous(),
            pos,
            ctx.children,
            dir,
            disp,
            grid_scales,
            depths,
            resolutions,
            entries,
            ctx.hashmap_size,
            features
        )
        grad_features = list(grads[:-1])
        grad_disp = grads[-1]

        return (
            None,
            None,
            None,
            None,
            grad_disp,
            None,
            None,
            None,
            None,
            None,
            *grad_features
        )


class OctaOctreeEncoding(nn.Module):
    """
    Octahedral-Octree feature encoding as graphics primitive.
    """
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
    _big_primes = torch.tensor(
        [1, 19349663, 83492791],
        dtype=torch.int64
    )

    def __init__(self, config: dict, bbox: torch.Tensor, points: torch.Tensor = None) -> None:
        super(OctaOctreeEncoding, self).__init__()
        self.config = config

        assert config["n_dims_to_encode"] == 5, "Octahedral-Octree encoding only supports 5D"
        # self.L_MAX = config["n_depths"]
        # self.L_MIN = config["full_depths"]
        self.full_depth = 2
        self.depths = config["octree_depths"]
        self.hashmap_size = 2 ** config["log2_hashmap_size"]
        self.D = config["n_features_per_sa"]

        self.resolutions = [int(2 ** i) for i in config["octa_resolutions"]]
        # self.resolutions.reverse()
        grid_scales = [(bbox[1] - bbox[0]) / (2 ** i) for i in self.depths]

        if points is not None:
            points = Points(points=points)
            self.octree = Octree(self.depths[-1], self.full_depth, device=points.device)
            self.octree.build_octree(points)
            self._update_octree_structure()
        else:
            self.octree = Octree(self.depths[-1], self.full_depth)

        # self.dump_memory_info()
        # exit()
        
        self.register_buffer("grid_scales", torch.stack(grid_scales, dim=0))
        self.register_buffer("spatial_offset", OctaOctreeEncoding._spatial_offset)

    def to(self, device: torch.device) -> "OctaOctreeEncoding":
        super().to(device)
        self.octree = self.octree.to(device)
        return self
    
    def _update_octree_structure(self):
        self.features = nn.ParameterList()
        # Update octree sizes
        for depth in range(self.octree.depth + 1):
            self.octree.nnum[depth] = self.octree.children[depth].shape[0]
            self.octree.nnum_nempty[depth] = self.octree.nempty_index(depth).shape[0]
        # Update OctreeEncoding sizes and features
        self.sizes = []
        self.entries = []
        for i, depth in enumerate(self.depths):
            size = self.octree.nempty_index(depth).shape[0]
            self.sizes.append(size)
            entries = size * (self.resolutions[i] + 1) ** 2
            self.entries.append(entries)
            if entries > self.hashmap_size:
                self.features.append(nn.Parameter(torch.zeros(self.hashmap_size, self.D)))
            else:
                self.features.append(nn.Parameter(torch.zeros(size, (self.resolutions[i] + 1) ** 2, self.D)))
        # Construct octree neighbors for interpolation
        self.octree.construct_all_neigh()

    def dump_memory_info(self):
        print("OctaOctreeEncoding Memory Info:")
        print(f"  Depths: {self.depths}")
        print(f"  Resolutions: {self.resolutions}")
        print(f"  Feature Dimension: {self.D}")
        print(f"  Octree Sizes: {self.sizes}")
        print(f"  Level Entries: {self.entries}")
        print(f"  Theoretical Entries: {sum(self.entries)}")
        total_memory = sum(p.element_size() * p.numel() for p in self.parameters())
        print(f"  Total Memory: {total_memory / (1024 ** 2):.2f} MB")
    
    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # Save octree's keys, children and neighs
        super()._save_to_state_dict(destination, prefix, keep_vars)

        for i, key in enumerate(self.octree.keys):
            if key is not None:
                destination[prefix + f"octree.keys.{i}"] = key if keep_vars else key.detach()

        for i, child in enumerate(self.octree.children):
            if child is not None:
                destination[prefix + f"octree.children.{i}"] = child if keep_vars else child.detach()
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # Load octree's keys, children and neighs
        device = self.octree.device
        self.octree.reset()

        for i in range(len(self.octree.keys)):
            key_name = prefix + f"octree.keys.{i}"
            if key_name in state_dict:
                self.octree.keys[i] = state_dict.pop(key_name).to(device)
            elif strict and self.octree.keys[i] is not None:
                missing_keys.append(key_name)

        for i in range(len(self.octree.children)):
            child_name = prefix + f"octree.children.{i}"
            if child_name in state_dict:
                self.octree.children[i] = state_dict.pop(child_name).to(device)
            elif strict and self.octree.children[i] is not None:
                missing_keys.append(child_name)
        
        self._update_octree_structure()

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
    
    @property
    def output_dim(self) -> int:
        if self.config["level_reduce"] == "Concat":
            return len(self.depths) * self.D
        return self.D
    
    def load_kernel(self, kernel_name: str = "octa_octree_cuda") -> None:
        """
        Load CUDA kernel for octahedral-octree.
        """
        if getattr(self, "cuda_kernel", None) is not None and getattr(self, "cuda_kernel_name", None) == kernel_name:
            self.use_kernel = True
            return

        self.use_kernel = True
        self.cuda_kernel_name = kernel_name

        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.cuda_kernel = load(
            name=kernel_name,
            sources=[
                os.path.join(current_dir, "octa_octree_cuda", "octa_octree_bindings.cpp"),
                os.path.join(current_dir, "octa_octree_cuda", "octa_octree_cuda.cu")],
            extra_cflags=[
                '-O3',
                "-DLEVELS={}".format(len(self.depths)),
                "-DL_MIN={}".format(self.full_depth),
                "-DDIMENSIONS={}".format(self.D),
                "-DDIMENSIONS_OUT={}".format(self.D),
                "-DLAYER_REDUCE={}".format(self.config["level_reduce"].upper()),
            ],
            extra_cuda_cflags=[
                "-O3", "-g", "-lineinfo", "-Xcompiler", "-rdynamic",
                "-DLEVELS={}".format(len(self.depths)),
                "-DL_MIN={}".format(self.full_depth),
                "-DDIMENSIONS={}".format(self.D),
                "-DDIMENSIONS_OUT={}".format(self.D),
                "-DLAYER_REDUCE={}".format(self.config["level_reduce"].upper()),
            ],
            verbose=True,
        )
    
    def forward(self, pos: torch.Tensor, dir: torch.Tensor, disps: torch.Tensor) -> torch.Tensor:
        if self.use_kernel:
            children = []
            for depth in range(self.full_depth, self.depths[-1] + 1):
                children.append(self.octree.children[depth].contiguous())
            for i in range(len(self.depths)):
                self.features[i] = self.features[i].contiguous()
            grid_scales = self.grid_scales.to(device=pos.device, dtype=torch.float32).contiguous()
            depths = torch.tensor(self.depths, device=pos.device, dtype=torch.int32).contiguous()
            resolutions = torch.tensor(self.resolutions, device=pos.device, dtype=torch.int32).contiguous()
            entries = torch.tensor(self.entries, device=pos.device, dtype=torch.int64).contiguous()

            result = _OctaOctreeQuery.apply(
                self.cuda_kernel,
                pos.contiguous(),
                children,
                dir.contiguous(),
                disps.contiguous(),
                grid_scales,
                depths,
                resolutions,
                entries,
                self.hashmap_size,
                *self.features
            )

            return result
        
        results = []
        for i, depth in enumerate(self.depths):
            # [N, 3]
            xyz = pos * (2 ** depth) - 0.5
            xyzi = xyz.floor()
            xyzf = xyz - xyzi
            # [N, 8, 3] -> [8N, 3]
            xyzn = (xyzi.unsqueeze(-2) + self.spatial_offset).view(-1, 3)
            # xyzn = (xyzi.unsqueeze(-2).repeat(1, 8, 1)).view(-1, 3)
            xyz_idx = self.octree.search_xyzb(
                torch.cat([xyzn, torch.zeros_like(xyzn[:, 0:1])], dim=1),
                depth, nempty=True
            ).reshape(-1, 8)
            # [N, 8]
            xyz_weight = ((1.0 - self.spatial_offset) - xyzf.unsqueeze(-2)).prod(dim=-1).abs()
            # Set invalid index and weight to 0
            xyzn = xyzn.view(-1, 8, 3)
            invalid = (xyzn < 0).any(dim=-1) | (xyzn >= 2 ** depth).any(dim=-1) | (xyz_idx < 0)
            xyz_weight[invalid] = 0.0
            xyz_idx[invalid] = 0
            xyz_weight = xyz_weight / (xyz_weight.sum(dim=-1, keepdim=True) + 1e-8)

            # [N] or [N, 1] shares one disparity across levels; [N, L] uses per-level disparity.
            if disps.dim() == 1:
                disp = disps[:, None]
            elif disps.shape[1] == 1:
                disp = disps
            else:
                disp = disps[:, i:(i+1)]
            # [N, 8, 3]
            perturb = (xyzf.unsqueeze(-2) - self.spatial_offset) * self.grid_scales[i]
            dir_pert = dir.unsqueeze(-2) + perturb * disp.unsqueeze(-2)
            sample_dir = torch.where(
                (disp.isinf() | disp.isnan())[..., None, :],
                dir.unsqueeze(-2),
                dir_pert / (torch.norm(dir_pert, dim=-1, keepdim=True) + 1e-8)
            )

            # Calculate octahedral coordinate
            # [N, 8, 2]
            uv = octa_dir_to_uv(sample_dir)
            resolution = self.resolutions[i]
            # Calculate directional base and offset
            # [N, 8, 2]
            uvi = torch.floor(uv * (resolution // 2)).to(torch.int64)
            uvf = uv * (resolution // 2) - uvi

            # Directional index
            # [N, 8, 3, 2] / [N, 8, 3]
            dir_idx, dir_weight = octa_lerp(uvi, uvf, resolution)
            # dir_idx = (uvi).unsqueeze(-2).repeat(1, 1, 3, 1)
            # dir_weight = torch.ones(dir_idx.shape[:-1], device=dir_idx.device) / 3.0

            # [N, 8, 3]
            if self.entries[i] > self.hashmap_size:
                index = OctaOctreeEncoding._big_primes[0].item() * xyz_idx.unsqueeze(-1) + \
                        OctaOctreeEncoding._big_primes[1].item() * dir_idx[..., 0] + \
                        OctaOctreeEncoding._big_primes[2].item() * dir_idx[..., 1]
                index = index % self.hashmap_size
            else:
                dir_idx = dir_idx[..., 0] * (resolution + 1) + dir_idx[..., 1]
                index = (xyz_idx * (resolution + 1) ** 2).unsqueeze(-1) + dir_idx
            weight = xyz_weight.unsqueeze(-1) * dir_weight

            # [N, 8, 3, D]
            result = self.features[i].reshape(-1, self.D)[index.view(-1)].view(-1, 8, 3, self.D)
            result = (result * weight.unsqueeze(-1)).sum(dim=(-3,-2))

            results.append(result)

        if self.config["level_reduce"] == "Concat":
            result = torch.cat(results, dim=-1)
        elif self.config["level_reduce"] == "Mean":
            result = torch.stack(results, dim=-1).mean(dim=-1)
        else:
            raise NotImplementedError("Unknown level reduce method: " + self.config["level_reduce"])
        
        return result
