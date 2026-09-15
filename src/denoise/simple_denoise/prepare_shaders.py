import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(CURRENT_DIR, "../../"))
sys.path.append(os.path.join(CURRENT_DIR, "../"))

from denoise.config import _C as cfg
from denoise.config import check_user_settings

from pyNISConfigWrapper import pyNISConfig

import shutil


def run_cmd(cmd: str) -> int:
    print("\033[93m[Info]\033[00m: run cmd:\n    {}".format(cmd))
    ret = os.system(cmd)
    if ret != 0:
        print("\033[91m    [Failed]\033[00m")
    else:
        print("\033[92m    [Success]\033[00m")
    return ret


def generate_OpenGL_shaders():

    vk_test_cmd = "glslc --version"
    in_path = run_cmd(vk_test_cmd) == 0
    shader_src_dir = os.path.join(CURRENT_DIR, "../shaders/NIS")
    shader_gen_dir = os.path.join(shader_src_dir, "generated/")

    if not in_path and cfg.VK_PATH == "":
        print("\033[93m[Warning]\033[00m: vulkan sdk not found, just use the default shader instead!")
        print("\033[93m[Warning]\033[00m: you can set the path of vulkan sdk in 'src/config.py', or you can add glslc to your PATH")
        print("\033[93m[Config]\033[00m: OptimalBlockWidth: 32, OptimalBlockHeight: 24, OptimalThreadGroupSize: 128")
        shutil.copyfile(os.path.join(shader_src_dir, "nis_scaler.comp"), os.path.join(shader_gen_dir, "nis_scaler.comp"))
        shutil.copyfile(os.path.join(shader_src_dir, "nis_sharpen.comp"), os.path.join(shader_gen_dir, "nis_sharpen.comp"))
        return

    # 1. get best config
    NIS_config = pyNISConfig()
    w, h, g = NIS_config.get_optimal_dispatch_size()
    # OptimalBlockWidth, OptimalBlockHeight, OptimalThreadGroupSize
    print("\033[93m[Info]\033[00m OptimalBlockWidth: {}, OptimalBlockHeight: {}, OptimalThreadGroupSize: {}".format(w, h, g))

    # 1. generate *.spv
    if (not os.path.exists(shader_gen_dir)):
        os.makedirs(shader_gen_dir)
    src_shader = os.path.join(shader_src_dir, "NIS_Main.glsl")
    spv_scaler = os.path.join(shader_gen_dir, "nis_scaler_glsl.spv")
    spv_sharpen = os.path.join(shader_gen_dir, "nis_sharpen_glsl.spv")
    glslc_exe = "glslc" if in_path else os.path.join(cfg.VK_PATH, "Bin/glslc")
    glslc_args = "-x glsl -DNIS_BLOCK_WIDTH={} -DNIS_THREAD_GROUP_SIZE={} -DNIS_USE_HALF_PRECISION=1 -DNIS_GLSL=1 -fshader-stage=comp".format(w, g)
    # scaler
    cmd = "{} -DNIS_SCALER=1 -DNIS_BLOCK_HEIGHT={} {} -o {} {}".format(glslc_exe, h, glslc_args, spv_scaler, src_shader)
    ret1 = run_cmd(cmd)
    # sharpen
    cmd = "{} -DNIS_SCALER=0 -DNIS_BLOCK_HEIGHT={} {} -o {} {}".format(glslc_exe, h, glslc_args, spv_sharpen, src_shader)
    ret2 = run_cmd(cmd)
    if (ret1 != 0 or ret2 != 0):
        print("\033[91m[Failed]\033[00m: generate *.spv failed!")
        return

    # 2. generate *.comp
    spirv_cross_exe = "spirv-cross" if in_path else os.path.join(cfg.VK_PATH, "Bin/spirv-cross")
    comp_scaler = os.path.join(shader_gen_dir, "nis_scaler_glsl.comp")
    comp_sharpen = os.path.join(shader_gen_dir, "nis_sharpen_glsl.comp")
    cmd = "{} --version 450 {} --output {}".format(spirv_cross_exe, spv_scaler, comp_scaler)
    ret1 = run_cmd(cmd)
    cmd = "{} --version 450 {} --output {}".format(spirv_cross_exe, spv_sharpen, comp_sharpen)
    ret2 = run_cmd(cmd)
    if (ret1 != 0 or ret2 != 0):
        print("\033[91m[Failed]\033[00m: generate *.comp failed!")
        return

    # 3.add some addtional info
    tonemap_file = os.path.join(shader_src_dir, "tonemap.glsl")
    tonemap_content = ""
    with open(tonemap_file, "r") as f:
        tonemap_content = f.read()

    # scale
    glsl_s = [comp_scaler, comp_sharpen]
    glsl_update_s = [
        os.path.join(shader_gen_dir, "nis_scaler.comp"),
        os.path.join(shader_gen_dir, "nis_sharpen.comp")
    ]
    for glsl, glsl_update in zip(glsl_s, glsl_update_s):
        with open(glsl, "r") as f:
            lines = f.readlines()
        res = []
        next_line_is_const = False
        const_buffer_name = ""
        for l in lines:
            if (next_line_is_const):
                const_buffer_name = l.strip('}; \n').strip()
                next_line_is_const = False
            if l.startswith("float getY(vec3 rgba)"):
                res.append("#define g_exposure {}.reserved0\n".format(const_buffer_name))
                res.append(tonemap_content)
            elif l.strip().startswith("float reserved1;"):
                next_line_is_const = True
            elif l.strip().startswith("imageStore"):
                res.append("op.xyz = tonemap_aces(op.xyz);\n".format(const_buffer_name))
            res.append(l)
        with open(glsl_update, "w") as f:
            f.writelines(res)
