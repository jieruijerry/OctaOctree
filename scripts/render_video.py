# Basics
import os
import json
import argparse
from tqdm import tqdm
import ffmpeg
import glfw

# Computational
import numpy as np
import torch

# Custom
from src.denoise.denoiser_wrap import *
from src.denoise.simple_denoise.filter import FilterTasks
from src.viewer.camera import FPSCamera, MovingCamera
from src.viewer.ui import UI
from render import load_render_vars

# Mitsuba
import drjit as dr
import mitsuba as mi
mi.set_variant("cuda_rgb")


def render_video(render_vars: dict, script: dict, args: argparse.Namespace) -> torch.Tensor:
    # Load render variables
    scene: mi.Scene = render_vars["scene"]
    params = mi.traverse(scene)

    # Check cache dir
    cache_dir = os.path.join("out", "video_cache")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    
    fps = script["fps"]
    duration = script["duration"]
    spp = script["spp"]
    render_mode = script["render_mode"]

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
    
    # Load moving camera
    cameras = []
    for camera_name in os.listdir(os.path.join("out", args.scene, "video_poses")):
        if not camera_name.endswith(".npz"):
            continue
        camera_path = os.path.join("out", args.scene, "video_poses", camera_name)
        with np.load(camera_path, allow_pickle=True) as data:
            extrinsics = data["extrinsics"]
            intrinsics = data["intrinsics"].item()
        cameras.append(FPSCamera(intrinsics, extrinsics, speed=1))
    cameras = MovingCamera(cameras)

    # Load denoiser
    if script["denoise"] == "Oidn":
        camera = render_vars["camera"]
        ui = UI(camera.width, camera.height, camera, bbox=scene.bbox())
        oidn_wrap = DenoiserWrap(scene=scene, ui=ui, type=5)
        oidn_wrap.type = 0  # DOIDN

    elif script["denoise"] == "FXAA":
        # opengl init
        if not glfw.init():
            print("Failed to initialize GLFW")
            exit()
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 6)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.VISIBLE, False)  # headless
        window = glfw.create_window(1, 1, "Off-Screen", None, None)
        if not window:
            print("Failed to create window")
            glfw.terminate()
            exit()
        glfw.make_context_current(window)

        camera = render_vars["camera"]
        fxaa_denoiser = FilterTasks(None, scene, camera.width, camera.height, [camera.width, camera.height])

    # Render
    cache_format = os.path.join("out", "video_cache", "{:02d}{:04d}.png")
    for section in range(len(cameras.cameras)):
        for frame in tqdm(range(duration * fps)):
            cache_path = cache_format.format(section, frame)
            # Skip if already rendered
            if os.path.exists(cache_path):
                continue

            # Interpolate camera
            cam_v = frame / (duration * fps)
            camera = cameras.get_camera(section, cam_v)
            params['PerspectiveCamera.to_world'] = mi.Matrix4f(camera.get_transform()[None, ...])
            params['PerspectiveCamera.x_fov'] = mi.Float32(camera.get_x_fov()[None, ...])
            params.update()

            # Render img
            seed = np.random.randint(0, 100000000)
            img = mi.render(scene, integrator=integrator, seed=seed, spp=spp)
            dr.sync_device()
            torch.cuda.synchronize()

            # Post-process
            if script["denoise"] == "Oidn":
                img = oidn_wrap.denoise(img)
            elif script["denoise"] == "FXAA":
                img_tensor: torch.Tensor = img.torch().cuda()
                # LHS with FXAA
                torch.cuda.synchronize()
                dr.sync_device()
                img = fxaa_denoiser.fetch_denoised_result_headless(img_tensor, True)
            
            mi.util.write_bitmap(cache_path, img)
            # print("Image saved to", output_path)
    
    # Convert to video
    imgs_path = os.path.join("out", "video_cache", "*.png")
    video_path = os.path.join("out", "{}.mp4".format(script["name"]))
    ffmpeg.input(imgs_path, pattern_type='glob', framerate=fps).output(
        video_path,
        vcodec='libx264',
        pix_fmt='yuv420p',
    ).run()
    print("Video saved to {}".format(video_path))


def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-S", "--script", type=str, default="nelp-oct")
    parser.add_argument("-c", "--config", type=str, default="nelp-oct")
    parser.add_argument("-s", "--scene", type=str, default="veach-ajar")
    parser.add_argument("-m", "--model_ckpt", type=str, default=None)
    parser.add_argument("-H", "--half_precision", type=bool, default=False)
    return parser.parse_args()


if __name__ == "__main__":
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    args = parse_args()
    
    # Load config file
    with open(os.path.join("configs", args.config + ".json"), "r") as f:
        config = json.load(f)
    
    # Load video script
    with open(os.path.join("configs", "video", args.script + ".json"), "r") as f:
        script = json.load(f)
    
    # Render
    render_vars = load_render_vars(config, args)
    render_video(render_vars, script, args)
    