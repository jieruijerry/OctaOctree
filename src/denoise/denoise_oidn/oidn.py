import setup_oidn_example as oidn
import torch
import drjit as dr
import mitsuba as mi
mi.set_variant("cuda_rgb")
import os
import imgui
from typing import Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


class OidnDenoiser:
    def name(self):
        return "[oidn]"

    def __init__(self, scene: mi.Scene):
        self.use_builtin_aov = False
        self.aux_integrator = None
        self.builtin_aux_integrator = None
        self.builtin_aux_integrator_inner = None

        self.scene: mi.Scene = scene
        self.sensor: mi.Sensor = scene.sensors()[0]

        oidn.init()

        self.aux = True

        # weight
        weights_files = os.listdir(os.path.join(CURRENT_DIR, "weights"))
        weights_files_names = [i.split(".")[0].split("_")[-1]
                               for i in weights_files]
        weights_files_names.insert(0, "None")
        # first no weight set
        self.oidn_weight_index = 0
        self.weights_files = weights_files
        self.weights_files_names = weights_files_names

    def denoise_simple(self, img: torch.tensor):
        img_clean = torch.zeros_like(img).to(img.device)
        oidn.denoise(img, img_clean, img.shape[1], img.shape[0], img.shape[2])
        return img_clean

    def denoise_albedo_normal(self, img_color: torch.tensor,
                              img_albedo: torch.tensor, img_normal: torch.tensor):
        img_clean = torch.zeros_like(img_color).to(img_color.device)
        oidn.denoise_with_normal_and_albedo(
            img_color, img_normal, img_albedo, img_clean,
            img_color.shape[1], img_color.shape[0], img_color.shape[2]
        )
        return img_clean

    def denoise(self, img: mi.TensorXf):
        ret = None
        if self.aux:
            # dr.sync_device() # wait for mi.render() to finish
            img_color = img[:, :, 0:3].torch()
            img_albedo = None
            img_normal = None
            if self.use_builtin_aov:
                img_albedo = img[:, :, 3:6].torch()
                img_normal = img[:, :, 6:9].torch()
            else:
                if (self.aux_integrator is None):
                    self.aux_integrator = mi.load_dict({'type': 'b_albedo_normal'})
                aux_tensor: mi.TensorXf = mi.render(
                    scene=self.scene, spp=1, seed=0, sensor=self.sensor, integrator=self.aux_integrator
                )
                img_albedo = aux_tensor[:, :, 0:3].torch()
                img_normal = aux_tensor[:, :, 3:6].torch()
            dr.sync_device()  # wait for 3 lines above to finish
            ret = self.denoise_albedo_normal(img_color, img_albedo, img_normal)
        else:
            img = img.torch()
            dr.sync_device()
            ret = self.denoise_simple(img)
        return ret

    def set_weights(self, weights_path: str):
        oidn.set_weights(weights_path)

    def render_ui(self, integrator: mi.Integrator) -> Tuple[bool, mi.Integrator]:
        value_changed = False

        # weight
        vc, self.oidn_weight_index = imgui.combo("{} Weights".format(self.name()),
                                                 self.oidn_weight_index, self.weights_files_names)
        if vc:
            weight_path = ""
            if self.oidn_weight_index != 0:
                weight_path = os.path.join(CURRENT_DIR, "weights", self.weights_files[self.oidn_weight_index - 1])
            self.set_weights(weight_path)
        value_changed = value_changed or vc

        # aux
        if self.oidn_weight_index == 0:
            vc, self.aux = imgui.checkbox("{} Use Albedo and Normal".format(self.name()), self.aux)
            value_changed = value_changed or vc
        else:
            # weight file indicates whether to use albedo and normal
            tmp = self.aux
            self.aux = self.weights_files[self.oidn_weight_index - 1].find("alb_nrm") != -1
            vc = tmp != self.aux
            value_changed = value_changed or vc

        if (self.aux):
            vc, self.use_builtin_aov = imgui.checkbox("{} Use Built-in Interator(slow)".format(self.name()), self.use_builtin_aov)
            value_changed = value_changed or vc

            if (vc):
                self.builtin_aux_integrator = None
                self.builtin_aux_integrator_inner = None

            if self.use_builtin_aov:
                if (self.builtin_aux_integrator is not None and self.builtin_aux_integrator_inner == integrator):
                    integrator = self.builtin_aux_integrator
                else:
                    print("{} reconstructing aux integrator".format(self.name()))
                    self.builtin_aux_integrator_inner = integrator
                    self.builtin_aux_integrator = mi.load_dict({
                        'type': 'aov',
                        'aovs': "albedo:albedo,sh_normal:sh_normal",
                        'integrator': integrator
                    })
                    integrator = self.builtin_aux_integrator

        return value_changed, integrator

    def free(self):
        oidn.free_denoiser()
        torch.cuda.empty_cache()
