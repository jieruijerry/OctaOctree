import numpy as np
import torch
from torch.utils.data import Dataset

import os


class CameraDataset(Dataset):
    def __init__(self, poses_dir: str):
        """
        Camera pose dataset.

        Args:
            poses_dir (str): Directory containing camera poses.
        """

        self.poses = []
        self.size = 0

        # Load camera poses from the specified directory
        if not os.path.exists(poses_dir):
            raise FileNotFoundError(f"Poses directory {poses_dir} does not exist.")
        
        print("Loading camera poses from", poses_dir)
        for pose_file in sorted(os.listdir(poses_dir)):
            if pose_file.endswith('.npz'):
                pose_path = os.path.join(poses_dir, pose_file)
                pose = np.load(pose_path, allow_pickle=True)
                pose = {
                    "extrinsics": pose["extrinsics"],
                    "intrinsics": {
                        "width": int(pose["intrinsics"].item()["width"]),
                        "height": int(pose["intrinsics"].item()["height"]),
                        "x_fov": float(pose["intrinsics"].item()["x_fov"]),
                    },
                }
                self.poses.append(pose)
                self.size += 1
        print("Total poses loaded:", self.size)

    def __len__(self):
        return self.size
    
    def __getitem__(self, idx: int):
        return self.poses[idx]


class DummyDataset(Dataset):
    def __init__(self, size: int = 1):
        self.size = size

        print("Using DummyDataset with size", self.size)

    def __len__(self):
        return self.size
    
    def __getitem__(self, idx: int):
        # Lightning skips batches that are literally `None`.
        # Return a harmless placeholder instead.
        return {"dummy": True}
