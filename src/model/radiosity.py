import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import tinycudann as tcnn

from src.module.basic import ShallowMLP
from src.module.hash_grid import HashGrid
from src.module.k_planes import KPlanes, PVPlanes
from src.module.octa_grid import OctaGrid
from src.module.octree import OctreeEncoding
from src.module.octa_octree import OctaOctreeEncoding
from src.sample.lhs_rhs import LHSRHS, extract_input, get_wr_itsc
from src.sample.util import *

import drjit as dr
import mitsuba as mi
mi.set_variant("cuda_rgb")


def get_time():
    dr.sync_device()
    torch.cuda.synchronize()
    return time.time()


class RadiosityPipeline(L.LightningModule):
    """
    Radiosity pipeline
    """
    def __init__(self, config: dict, scene: mi.Scene) -> None:
        super(RadiosityPipeline, self).__init__()
        self.config = config
        self.scene = scene
        self.register_buffer("bbox", get_model_bbox(scene))

    def training_step(self, batch, batch_idx):
        """
        Training step for the model
        """
        seed = self.global_step * self.trainer.world_size + self.global_rank

        # Adaptive RHS and epsilon
        ad_ratio = 2 ** int(4 * (self.global_step / self.trainer.max_steps))
        point_num = self.config["sample"]["n_points"] // ad_ratio
        dirs_per_point = self.config["sample"]["n_dirs_per_point"] * ad_ratio
        eps = 1e-1 / ad_ratio

        # Sample
        lhs_rhs = LHSRHS(
            scene=self.scene,
            point_num=point_num,
            dirs_per_point=dirs_per_point,
        )
        if self.config["sample"]["from_poses"]:
            lhs_rhs.sample(seed=seed, pose=batch)
        else:
            lhs_rhs.sample(seed=seed)
        lhs_rhs.to(device=self.device)

        # Forward pass
        result = self(lhs_rhs)
        lhs_color = result["lhs"]
        rhs_color = result["rhs"].detach()

        # Compute loss
        nr_norm = (rhs_color + lhs_color).detach() / 2 + eps
        loss = torch.mean(((rhs_color - lhs_color) / nr_norm) ** 2) / ad_ratio

        if result["reg"] is not None:
            reg_loss = torch.mean(result["reg"])
            loss += self.config["train"]["reg_lambda"] * reg_loss
            self.log("reg_loss", reg_loss.item(), prog_bar=True)

        # Logging
        self.log("loss", loss.item(), prog_bar=True)
        self.log("real_step", self.global_step, prog_bar=True)

        if (self.global_step + 1) % 200 == 0 and self.global_rank == 0:
            print(f"##### Step {self.global_step + 1} #####")
            print("lhs color", lhs_color[:3])
            print("rhs color", rhs_color[:3])

        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config["train"]["learning_rate"])
    
    def forward(self, lhs_rhs: LHSRHS) -> torch.Tensor:
        """
        Query the model with lhs and rhs interactions
        """
        si_lhs = lhs_rhs.si_lhs
        si_rhs = lhs_rhs.si_bsdf

        lhs_color, reg = self.query_model(si_lhs)
        with torch.no_grad():
            rhs_color, _ = self.query_model(si_rhs)

        # Render rhs color
        rhs_color = rhs_color.reshape(-1, lhs_rhs.dirs_per_point, 3)
        rhs_color = lhs_rhs.render(rhs_color, None)

        return {
            "lhs": lhs_color,
            "rhs": rhs_color,
            "reg": reg,
        }
    
    def query_model(self, si: mi.SurfaceInteraction3f) -> torch.Tensor:
        """
        Query the model with surface interaction
        """
        raise NotImplementedError("RadiosityPipeline is an abstract class, please implement query_model method in the subclass.")
    
    def render_lhs(self, si_lhs: mi.SurfaceInteraction3f, precision=torch.float32):
        with torch.no_grad():
            lhs_color, _ = self.query_model(si_lhs, precision=precision)
        
        return lhs_color
    
    def render_rhs(self, si_lhs: mi.SurfaceInteraction3f, precision=torch.float32, spp: int = 1):
        with torch.no_grad():
            point_num = si_lhs.p.torch().shape[0]
            result = torch.zeros((point_num, 3), dtype=precision, device=self.device)

            render_iter = (spp + 3) // 4
            for i in range(render_iter):
                iter_spp = min(4, spp - i * 4)
                # Sample rhs interactions
                lhs_rhs = LHSRHS(
                    scene=self.scene,
                    point_num=point_num,
                    dirs_per_point=iter_spp
                )
                lhs_rhs.sample(seed=np.random.randint(0, 1000000), si_lhs=si_lhs)
                lhs_rhs.to(device=self.device, dtype=precision)
                si_rhs = lhs_rhs.si_bsdf

                rhs_color, _ = self.query_model(si_rhs, precision=precision)

                # Render rhs color
                rhs_color = rhs_color.reshape(-1, iter_spp, 3)
                rhs_color = lhs_rhs.render(rhs_color, None)

                result += rhs_color * (iter_spp / spp)

                dr.sync_device()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

        return result
    

class NeuralRadiosity(RadiosityPipeline):
    """
    Neural Radiosity model
    """
    def __init__(self, pipeline_config: dict, scene: mi.Scene) -> None:
        super(NeuralRadiosity, self).__init__(pipeline_config, scene)

        config = pipeline_config["model"]["ray"]

        if config["name"] == "HashGrid":
            self.encoding = HashGrid(config)
        elif config["name"] == "KPlanes":
            self.encoding = KPlanes(config)
        elif config["name"] == "CFKGrid":
            self.encoding = OctaGrid(config)
        elif config["name"] == "Octree":
            # Sample points from the scene, normalize to [-1, 1]
            points = sample_si(scene, 2 ** 26).p.torch()
            points = normalize_pos(points, self.bbox.to(device=points.device)) * 2 - 1
            # Initialize octree encoding
            self.encoding = OctreeEncoding(config, points)
        else:
            raise NotImplementedError()

        if config["use_tcnn"]:
            network_config = {
                "otype": "FullyFusedMLP" if config["n_hidden_dims"] <= 128 else "CutlassMLP",
                "n_hidden_layers": config["n_hidden_layers"],
                "n_neurons": config["n_hidden_dims"],
                "activation": "ReLU",
                "output_activation": config["output_activation"],
            }
            self.mlp = tcnn.Network(
                n_input_dims=self.encoding.output_dim + 3 * 3 + 1 + config["n_dims_to_encode"],
                n_output_dims=3,
                network_config=network_config
            )
        else:
            self.mlp = ShallowMLP(
                # encoding + pos + normal + wr + albedo + roughness
                in_channels=self.encoding.output_dim + 3 * 3 + 1 + config["n_dims_to_encode"],
                out_channels=3,
                hidden_layers=config["n_hidden_layers"],
                hidden_channels=config["n_hidden_dims"],
                activation=nn.ReLU(),
                output_activation=nn.Identity() if config["output_activation"] == "None" else nn.Softplus()
            )

    def train(self, mode: bool = True):
        """
        Set the model to training mode
        """
        super().train(mode)
        # if not mode:
        #     self.hash_grid.load_kernel("NR_hash_grid_cuda")
        # else:
        #     self.hash_grid.use_kernel = False
        return self

    def query_model(self, si: mi.SurfaceInteraction3f, precision=torch.float32) -> torch.Tensor:
        """
        Query the model with surface interaction
        """
        dr.eval(si)
        pos, normal, dir, albedo, roughness, active_side = extract_input(si, device=self.device, dtype=precision)

        # Hash grid encoding
        pos = normalize_pos(pos, self.bbox)
        enc = self.encoding(pos)

        # Concatenate encoding with wr_direction and roughness
        enc = torch.cat([enc, pos, dir, normal, albedo, roughness], dim=-1)

        # Pass through MLP
        color = torch.abs(self.mlp(enc)).to(dtype=precision)

        return color, None


class NeuralLightProbe(RadiosityPipeline):
    """
    Neural Octahedron Radiosity model
    """
    def __init__(self, pipeline_config: dict, scene: mi.Scene) -> None:
        super(NeuralLightProbe, self).__init__(pipeline_config, scene)

        config = pipeline_config["model"]

        diff_cfg = config["diff"]
        spec_cfg = config["spec"]
        mlp_cfg = config["mlp"]

        mlp_cfg["otype"] = "FullyFusedMLP" if mlp_cfg["n_neurons"] <= 128 else "CutlassMLP"

        self.pos_enc = HashGrid(diff_cfg)
        self.dir_enc = OctaGrid(spec_cfg, bbox=self.bbox)

        self.depth_mlp = tcnn.Network(
            n_input_dims=self.pos_enc.output_dim + 3 + 1,
            n_output_dims=1,
            network_config={
                "otype": "FullyFusedMLP",
                "n_hidden_layers": 2,
                "n_neurons": 64,
                "activation": "ReLU",
                "output_activation": "SquarePlus"
            }
        )

        self.mlp = tcnn.Network(
            n_input_dims=self.pos_enc.output_dim + self.dir_enc.output_dim + 3 * 4 + 1,
            n_output_dims=3,
            network_config=mlp_cfg
        )
    
    def query_model(self, si, precision=torch.float32):
        """
        Query the model with surface interaction
        """
        # Trace specular reflection depth
        si_wr = get_wr_itsc(si, self.scene)
        wr_disp_gt = (1 / si_wr.t).torch()[..., None]

        # Extract input features
        dr.eval(si)
        pos, normal, dir, albedo, roughness, active_side = extract_input(si, device=self.device, dtype=precision)

        # Positional encoding
        pos_normalized = normalize_pos(pos, self.bbox)
        pos_enc = self.pos_enc(pos_normalized)

        # Disparity estimation
        wr_disp = self.depth_mlp(torch.cat([pos_enc, dir, roughness], dim=-1))

        # Directional encoding
        dir_enc = self.dir_enc(pos_normalized, dir, wr_disp)

        # Concatenate encoding with wr_direction and roughness
        enc = torch.cat([pos_enc, dir_enc, pos, dir, normal, albedo, roughness], dim=-1)

        # Pass through MLP
        color = torch.abs(self.mlp(enc)).to(dtype=precision)

        return color, torch.abs((wr_disp - wr_disp_gt) / (wr_disp_gt + 1e-2))


class OctreeNeLP(RadiosityPipeline):
    """
    Octree-based neural light probe model
    """
    def __init__(self, pipeline_config: dict, scene: mi.Scene) -> None:
        super(OctreeNeLP, self).__init__(pipeline_config, scene)

        config = pipeline_config["model"]

        disp_cfg = config["disparity"]
        enc_cfg = config["encoding"]
        mlp_cfg = config["mlp"]
        mlp_cfg["otype"] = "FullyFusedMLP" if mlp_cfg["n_neurons"] <= 128 else "CutlassMLP"

        self.disp_net = tcnn.NetworkWithInputEncoding(
            n_input_dims=3+3+1,
            n_output_dims=1,
            encoding_config=disp_cfg,
            network_config={
                "otype": "FullyFusedMLP",
                "n_hidden_layers": 2,
                "n_neurons": 64,
                "activation": "ReLU",
                "output_activation": "SquarePlus"
            }
        )
        # Sample points from the scene, normalize to [-1, 1]
        points = sample_si(scene, 2 ** 26).p.torch()
        points = normalize_pos(points, self.bbox.to(device=points.device)) * 2 - 1
        # Initialize octree encoding
        self.encoding = OctaOctreeEncoding(enc_cfg, bbox=self.bbox, points=points)

        self.mlp = tcnn.Network(
            n_input_dims=self.encoding.output_dim + 3 * 4 + 1,
            n_output_dims=3,
            network_config=mlp_cfg
        )

    def train(self, mode: bool = True):
        """
        Set the model to training mode
        """
        super().train(mode)
        self.encoding.load_kernel("octa_octree_encoding_cuda")
        return self
    
    def query_model(self, si, precision=torch.float32):
        """
        Query the model with surface interaction
        """
        # Trace specular reflection depth
        # si_wr = get_wr_itsc(si, self.scene)
        # wr_disp_gt = torch.nan_to_num((1 / si_wr.t).torch()[..., None], nan=0.0, posinf=0.0, neginf=0.0)

        # Extract input features
        dr.eval(si)
        pos, normal, dir, albedo, roughness, active_side = extract_input(si, device=self.device, dtype=precision)
        dir = torch.where(roughness < 0.5, dir, normal)

        # Disparity estimation
        wr_disp = self.disp_net(torch.cat([pos, dir, roughness], dim=-1))
        # return torch.where(roughness < 0.5, wr_disp.repeat(1, 3), torch.zeros_like(dir)), None

        # Radiance encoding
        pos_normalized = normalize_pos(pos, self.bbox)
        enc = self.encoding(pos_normalized, dir, wr_disp.to(torch.float32))
        # enc = enc.view(-1, 8, 4)
        # return torch.sigmoid(10 * enc[:, 7, :3]), None
        # enc = enc.view(-1, self.encoding.output_dim)

        # Concatenate encoding with wr_direction and roughness
        enc = torch.cat([enc, pos, dir, normal, albedo, roughness], dim=-1)

        # Pass through MLP
        color = torch.abs(self.mlp(enc)).to(dtype=precision)

        return color, None # torch.abs(1e1 * (wr_disp - wr_disp_gt) / (wr_disp_gt + 1e1)) * (1 - roughness)
