import torch
import torch.nn as nn

import ocnn
from ocnn.octree import Octree, Points


class OctreeEncoding(nn.Module):
    """
    Octree feature encoding as graphics primitive.
    """
    def __init__(self, config: dict, points: torch.Tensor = None) -> None:
        super(OctreeEncoding, self).__init__()
        self.config = config

        assert config["n_dims_to_encode"] == 3, "Octree encoding only supports 3D"
        self.L_MAX = config["n_depths"]
        self.L_MIN = config["full_depths"]
        self.D = config["n_features_per_level"]

        if points is not None:
            points = Points(points=points)
            self.octree = Octree(self.L_MAX, self.L_MIN, device=points.device)
            self.octree.build_octree(points)
            self._update_octree_structure()
        else:
            self.octree = Octree(self.L_MAX, self.L_MIN)

    def to(self, device: torch.device) -> "OctreeEncoding":
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
        for i, depth in enumerate(range(self.L_MIN, self.L_MAX + 1)):
            size = self.octree.nempty_index(depth).shape[0]
            self.sizes.append(size)
            self.features.append(nn.Parameter(torch.zeros(size, self.D)))
        # Construct octree neighbors for interpolation
        self.octree.construct_all_neigh()
    
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

        state_dict.pop("encoding.sizes", None)

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
    
    @property
    def output_dim(self) -> int:
        if self.config["level_reduce"] == "Concat":
            return (self.L_MAX - self.L_MIN + 1) * self.D
        return self.D
    
    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        features = []
        for i, depth in enumerate(range(self.L_MIN, self.L_MAX + 1)):
            xyz = pos * (2 ** depth)
            xyzb = torch.cat([xyz, torch.zeros_like(xyz[..., 0:1])], dim=-1)

            if self.config["interpolation"] == "Linear":
                feature = ocnn.nn.octree_linear_pts(self.features[i], self.octree, depth, xyzb, nempty=True)
            else:
                feature = ocnn.nn.octree_nearest_pts(self.features[i], self.octree, depth, xyzb, nempty=True)

            features.append(feature)

        features = torch.cat(features, dim=-1)
        return features
