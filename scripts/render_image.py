# Basics
import os
import json
import argparse
from tqdm import tqdm
import ffmpeg

# Computational
import numpy as np
import torch

# Custom
from src.viewer.camera import FPSCamera, MovingCamera
from src.integrator.neural import RadiosityIntegrator
# from src.model.radiosity import DNRPipeline
from render import load_render_vars

# Mitsuba
import drjit as dr
import mitsuba as mi
mi.set_variant("cuda_rgb")


def render_image(render_vars: dict, args: argparse.Namespace) -> torch.Tensor:
    # Load render variables
    scene: mi.Scene = render_vars["scene"]
    params = mi.traverse(scene)
    
    spp = args.spp
    render_mode = args.render_mode

    if render_mode == "LHS":
        integrator = render_vars["integrators"][args.config]
        integrator.render_mode = "LHS"
        spp = 1
    elif render_mode == "RHS":
        integrator = render_vars["integrators"][args.config]
        integrator.render_mode = "RHS"
        integrator.spp = spp
        spp = 1
    elif render_mode == "PT":
        integrator = render_vars["integrators"]["path"]
    else:
        raise ValueError("Invalid render mode:", render_mode)
    
    # Load camera
    with np.load(os.path.join("out", args.scene, "camera.npz"), allow_pickle=True) as data:
        extrinsics = data["extrinsics"]
        intrinsics = data["intrinsics"].item()
    
    camera = FPSCamera(intrinsics, extrinsics, speed=1)
    params['PerspectiveCamera.to_world'] = mi.Matrix4f(camera.get_transform()[None, ...])
    params['PerspectiveCamera.x_fov'] = mi.Float32(camera.get_x_fov()[None, ...])
    params.update()
    
    # Render
    max_spp_per_iter = 1024
    iter_num = (spp + max_spp_per_iter - 1) // max_spp_per_iter
    spp = spp if iter_num == 1 else max_spp_per_iter

    size = scene.sensors()[0].film().size()
    img: mi.TensorXf = dr.zeros(mi.TensorXf, (size[1], size[0], 3))
    
    for i in tqdm(range(iter_num)):
        img += mi.render(scene, integrator=integrator, spp=spp, seed=np.random.randint(0, 1000000))
        dr.flush_malloc_cache()
    img = img / iter_num
    
    # Save image
    image_path = os.path.join("out", "{}.exr".format(args.output))
    mi.util.write_bitmap(image_path, img)


def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="nelp-oct")
    parser.add_argument("-s", "--scene", type=str, default="veach-ajar")
    parser.add_argument("-r", "--render_mode", type=str, default="PT", choices=["LHS", "RHS", "PT"])
    parser.add_argument("-p", "--spp", type=int, default=102400)
    parser.add_argument("-o", "--output", type=str, default="test")
    parser.add_argument("-m", "--model_ckpt", type=str, default=None)
    parser.add_argument("-H", "--half_precision", type=bool, default=False)
    return parser.parse_args()


if __name__ == "__main__":
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    args = parse_args()
    
    # Load config file
    with open(os.path.join("configs", args.config + ".json"), "r") as f:
        config = json.load(f)
    
    # Render
    render_vars = load_render_vars(config, args)
    render_image(render_vars, args)