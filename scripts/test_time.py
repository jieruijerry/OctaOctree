import mitsuba as mi
import drjit as dr
import numpy as np
import torch
import json
import argparse
import os

from datetime import datetime
import time
from tqdm import trange

import glfw
# from src.denoise.function_wrap import load_camera_txt_and_set
# from src.denoise.denoiser_wrap import *
# from src.denoise.denoise_oidn.oidn import OidnDenoiser
# from src.denoise.simple_denoise.filter import FilterTasks

from render import load_render_vars
from src.viewer.camera import FPSCamera

mi.set_variant("cuda_rgb")


def sync():
    torch.cuda.synchronize()
    dr.sync_device()
    torch.cuda.synchronize()
    dr.sync_device()
    return time.time()


def schedule(integrator, scene, spp, thresh, seed):
    if (spp <= thresh):
        integrator.spp = spp
        return mi.render(scene, integrator=integrator, spp=1, seed=seed)

    spp_left = spp
    img = None
    r_seed = seed
    while (spp_left > 0):
        spp_render = min(spp_left, thresh)
        integrator.spp = spp_render
        img_rhs = mi.render(scene=scene, integrator=integrator, spp=1, seed=r_seed)
        if (img is None):
            img = img_rhs
        else:
            img = img + img_rhs * spp_render
        spp_left -= spp_render
        r_seed = np.random.randint(0, 100000)
    img = img * (1 / spp)
    return img


def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="ncr")
    parser.add_argument("-s", "--scene", type=str, default="veach-ajar")
    parser.add_argument("-m", "--model_ckpt", type=str, default=None)
    parser.add_argument("-o", "--output", type=str, default="./out/test.exr")
    parser.add_argument("-H", "--half_precision", type=bool, default=False)

    # if have --denoise, then args.denoise is True, else False
    parser.add_argument("--denoise", action="store_true")
    parser.add_argument("--box_filter", action="store_true")

    parser.add_argument("--save", action="store_true")
    parser.add_argument("-w", type=str, default="11110")  # means 5 works(1: do, 0: ignore)
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load config file
    with open(os.path.join("configs", args.config + ".json"), "r") as f:
        config = json.load(f)

    assert (len(args.w) == 5)
    # prepare scene

    # Load render variables
    render_vars = load_render_vars(config, args)
    scene: mi.Scene = render_vars["scene"]
    params: mi.SceneParameters = mi.traverse(scene)
    nr_integrator = render_vars["integrators"][args.config]
    path_integrator = render_vars["integrators"]["path"]
    camera: FPSCamera = render_vars["camera"]
    width, height = camera.width, camera.height

    # Load camera
    with np.load(os.path.join("out", args.scene, "camera.npz"), allow_pickle=True) as data:
        extrinsics = data["extrinsics"]
        x_fov = data["intrinsics"].item()["x_fov"]
        camera.set_transform(extrinsics)
        camera.set_x_fov(x_fov)
    print("Camera config loaded from", os.path.join("out", args.scene, "camera.npz"))

    params['PerspectiveCamera.to_world'] = mi.Matrix4f(camera.get_transform()[None, ...])
    params['PerspectiveCamera.x_fov'] = mi.Float32(camera.get_x_fov()[None, ...])
    params.update()

    if (args.box_filter):
        film: mi.Film = scene.sensors()[0].film()
        film.set_rfilter("box")

    # rendering
    # denoiser = OidnDenoiser(scene=scene)
    # denoiser.aux = True

    # fxaa
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

    # fxaa_denoiser = FilterTasks(None, scene, width, height, (width, height))

    timestamp = datetime.today().strftime('%Y-%m-%d-%H%M%S')

    img = None
    warm_iter = 10
    test_iter = 100
    pt_spp = 16
    odin_spp = 4
    lhs_spp = 1
    rhs_spp = 16
    lrhs_thresh = 8  # TODO: related to the Network Configuration
    info = []
    result = []
    r_seed = 1322

    # PT
    if (args.w[0] == "1"):
        name = "PT({}spp)".format(pt_spp)
        info.append(name)
        print(name)
        print("Warm up")
        for i in trange(warm_iter):
            img = mi.render(scene, integrator=path_integrator, spp=pt_spp, seed=r_seed)
        print("Test")
        a = sync()
        for i in trange(test_iter):
            img = mi.render(scene, integrator=path_integrator, spp=pt_spp, seed=r_seed)
        b = sync()
        c = (b - a) / test_iter
        print("Average Time: {}".format(c))
        result.append(c)
        if (args.save):
            mi.util.write_bitmap("screenshots/{}-pt-{}.exr".format(timestamp, pt_spp), img)

    # # denoise
    # if (args.w[1] == "1"):
    #     name = "Oidn({}spp)".format(odin_spp)
    #     info.append(name)
    #     print(name)
    #     print("Warm up")
    #     for i in trange(warm_iter):
    #         img = mi.render(scene, integrator=path_integrator, spp=odin_spp, seed=r_seed)
    #         img = denoiser.denoise(img)
    #     print("Test")
    #     a = sync()
    #     for i in trange(test_iter):
    #         img = mi.render(scene, integrator=path_integrator, spp=odin_spp, seed=r_seed)
    #         img = denoiser.denoise(img)
    #     b = sync()
    #     c = (b - a) / test_iter
    #     print("Average Time: {}".format(c))
    #     result.append(c)
    #     if (args.save):
    #         mi.util.write_bitmap("screenshots/{}-oidn-{}.exr".format(timestamp, odin_spp), img)

    # LHS
    if (args.w[2] == "1"):
        nr_integrator.render_mode = "LHS"
        name = "LHS({}spp)".format(lhs_spp)
        info.append(name)
        print(name)
        print("Warm up")
        for i in trange(warm_iter):
            img = schedule(nr_integrator, scene, lhs_spp, lrhs_thresh, r_seed)
        print("Test")
        a = sync()
        for i in trange(test_iter):
            img = schedule(nr_integrator, scene, lhs_spp, lrhs_thresh, r_seed)

        b = sync()
        c = (b - a) / test_iter
        print("Average Time: {}".format(c))
        result.append(c)
        if (args.save):
            mi.util.write_bitmap("screenshots/{}-lhs-{}.exr".format(timestamp, lhs_spp), img)

    # # LHS with FXAA
    # if (args.w[3] == "1"):
    #     nr_integrator.render_mode = "LHS"
    #     name = "LHS(+FXAA)({}spp)".format(lhs_spp)
    #     info.append(name)
    #     print(name)
    #     print("Warm up")
    #     for i in trange(warm_iter):
    #         img = schedule(nr_integrator, scene, lhs_spp, lrhs_thresh, r_seed)
    #         img = fxaa_denoiser.fetch_denoised_result_headless(img.torch(), False)
    #     print("Test")
    #     a = sync()
    #     for i in trange(test_iter):
    #         img = schedule(nr_integrator, scene, lhs_spp, lrhs_thresh, r_seed)
    #         img = fxaa_denoiser.fetch_denoised_result_headless(img.torch(), False)
    #     b = sync()
    #     c = (b - a) / test_iter
    #     print("Average Time: {}".format(c))
    #     result.append(c)
    #     if (args.save):
    #         # very slow to get texture from gpu to cpu(we do not need this step in practice infact)
    #         img = mi.render(scene, integrator=nr_integrator, spp=lhs_spp, seed=r_seed)
    #         img = fxaa_denoiser.fetch_denoised_result_headless(img.torch(), True)
    #         mi.util.write_bitmap("screenshots/{}-lhs-{}-fxaa.exr".format(timestamp, lhs_spp), img)

    # RHS
    if (args.w[4] == "1"):
        nr_integrator.render_mode = "RHS"
        name = "RHS({}spp)".format(rhs_spp)
        info.append(name)
        print(name)
        print("Warm up")
        for i in trange(warm_iter):
            img = schedule(nr_integrator, scene, rhs_spp, lrhs_thresh, r_seed)
        print("Test")
        a = sync()
        for i in trange(test_iter):
            img = schedule(nr_integrator, scene, rhs_spp, lrhs_thresh, r_seed)
        b = sync()
        c = (b - a) / test_iter
        result.append(c)
        print("Average Time: {}".format(c))
        if (args.save):
            mi.util.write_bitmap("screenshots/{}-rhs-{}.exr".format(timestamp, rhs_spp), img)

    print(result)
    for k, v in zip(info, result):
        print("{}: time = {}, fps = {}".format(k, v, 1 / v))
