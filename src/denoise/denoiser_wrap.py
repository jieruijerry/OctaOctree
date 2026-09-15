import os
import sys
import mitsuba as mi
import torch
import drjit as dr
import imgui
import numpy as np

# for utils
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DENOISE_ROOT_DIR = os.path.join(CURRENT_DIR, "..")
sys.path.append(CURRENT_DIR)
sys.path.append(DENOISE_ROOT_DIR)

from denoise.utils import empty_cache
# from denoise.cuda_extension.cuda_aa import CUDAAADenoiser

# for oidn module
BIN_DIR = ""
if sys.platform == "win32":
    BIN_DIR = "{}/denoise_oidn/oidn/oidn-2.1.0.x64.windows/bin".format(
        CURRENT_DIR)
elif sys.platform == "linux":
    BIN_DIR = "{}/denoise_oidn/oidn/oidn-2.1.0.x86_64.linux/lib".format(
        CURRENT_DIR)

if os.name == "nt":
    os.add_dll_directory(BIN_DIR)
elif os.name == "posix":
    # linux may not need following 3 lines
    os.environ["PATH"] += os.pathsep + BIN_DIR
    os.environ["LD_LIBRARY_PATH"] += os.pathsep + BIN_DIR
    sys.path.append(BIN_DIR)

from src.viewer.ui import UI

from denoise_oidn.oidn import OidnDenoiser
# if sys.platform == "win32":
from simple_denoise.filter import FilterTasks

TYPES = ["oidn", "Simple"]
OFFLINE_TYPES = []
ONLINE_TYPES_NUM = len(TYPES) - len(OFFLINE_TYPES)
# type
DOIDN = 0
DSIMPLE = 1
DTYPES = [DOIDN, DSIMPLE]
# platform
PWINDOWS = 0
PLINUX = 1


def platform_test():
    # 0: windows, 1: linux
    platform = ["windows", "linux"]
    ret = np.ones((len(TYPES), 2))
    ret[DSIMPLE][PLINUX] = 0
    print("Denoiser test on platform:")
    for i in range(len(TYPES)):
        for j in range(2):
            if (ret[i][j] != 0):
                print("  \033[0;32m[{}] Tested on {}\033[0m".format(TYPES[i], platform[j]))
            else:
                print("  \033[0;31m[{}] Not tested on {}\033[0m".format(TYPES[i], platform[j]))


class DenoiserWrap:
    def __init__(self, scene: mi.Scene, ui: UI, type=0):
        platform_test()

        self.ui = ui

        self.cuda_aa = None
        self.optix_denoiser_mi = None
        self.optix_denoiser = None
        self.odin_denoiser = None
        self.wskpd_denoiser = None
        self.afgsa_denoiser = None
        self.denoise_task: FilterTasks = None
        self.anf_denoiser = None

        self.wskpd_denoiser_model_path_index = 0
        self.wskpd_denoiser_use_new_aux = True

        self.scene = scene
        self.type = type

    def get_denoiser(self, type: int):
        if type == DOIDN:
            if self.odin_denoiser is None:
                self.odin_denoiser = OidnDenoiser(self.scene)
            return self.odin_denoiser
        elif type == DSIMPLE:
            if self.denoise_task is None:
                self.denoise_task = FilterTasks(self.ui, self.scene)
            self.ui.set_compute_task(self.denoise_task, False)
            return self.denoise_task
        else:
            return None

    def free_all_denoisers(self):
        self.optix_denoiser_mi = None
        if self.optix_denoiser is not None:
            self.optix_denoiser.free()
            self.optix_denoiser = None
        if self.odin_denoiser is not None:
            self.odin_denoiser.free()
            self.odin_denoiser = None
        self.wskpd_denoiser = None
        self.afgsa_denoiser = None
        if (self.denoise_task is not None):
            self.denoise_task.release()
            self.denoise_task = None
            self.ui.set_compute_task(None, False)
        self.cuda_aa = None
        empty_cache()

    def denoise(self, img: mi.TensorXf, skip_offline=True) -> torch.Tensor:
        if skip_offline and (self.type >= ONLINE_TYPES_NUM):
            print("\033[0;31m[DenoiserWrap]\033[0m offline denoiser: \033[0;31m{}\033[0m, skip".format(TYPES[self.type]))
            self.type = -1

        ret = None
        denoiser = self.get_denoiser(self.type)

        # denoise
        if self.type in DTYPES:
            ret = denoiser.denoise(img)
        else:
            ret = img.torch()
            if (img.shape[2] > 3):
                ret = ret[:, :, 0:3]
            # red error
            print("\033[0;31m[DenoiserWrap]\033[0m unknown denoiser type: \033[0;31m{}\033[0m, reset to {}".format(self.type, TYPES[0]))
            self.type = 0
        return ret

    def render_ui(self, integrator: mi.Integrator) -> mi.Integrator:
        value_changed = False
        if imgui.tree_node("Denoise Options", imgui.TREE_NODE_DEFAULT_OPEN):
            vc, self.type = imgui.combo("Denoiser Type", self.type, TYPES)
            if vc:
                self.free_all_denoisers()
            value_changed = value_changed or vc
            denoiser = self.get_denoiser(self.type)

            if (self.type == DOIDN):
                vc, integrator = denoiser.render_ui(integrator)
                value_changed = value_changed or vc
            elif (self.type == DSIMPLE):
                # TODO: same formulation as other denoisers
                vc, integrator = denoiser.render_ui(integrator)
                value_changed = value_changed or vc
            else:
                print("\033[0;31m[DenoiserWrap]\033[0m unknown denoiser type: \033[0;31m{}\033[0m, reset to default".format(
                    self.type))

            imgui.tree_pop()

        return value_changed, integrator

    def get_denoiser_type(self) -> str:
        return TYPES[self.type]

    def prepare_before_save(self, img: torch.tensor):
        return img

    def set(self, **kwargs):
        pass
        # example:
        # if self.denoise_task is not None:
        # self.denoise_task.set(**kwargs)

    def no_tonemapping(self):
        return self.denoise_task is not None
