import imgui
import torch
import mitsuba as mi
import numpy as np
import time
import os
from scripts.metrics import compute_metric_torch, compute_metric
from typing import Tuple
import drjit as dr

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_camera_txt_and_set(dscene, path):
    with open(path, "r") as f:
        strs = f.read().split("\n")
        camera_matrix_str = None
        for i in strs:
            i = i.strip()
            if i.startswith("#") or i == "":
                continue
            camera_matrix_str = i.strip().split(",")

        # strip '\t', ' '
        camera_matrix_str = [i.strip() for i in camera_matrix_str if i.strip()]
        # filter empty string
        camera_matrix_str = [i for i in camera_matrix_str if i]

    if len(camera_matrix_str) != 16:
        print("\033[0;31m[{}]\033[0m format error".format(path))
        return

    camera_matrix = [float(i) for i in camera_matrix_str]
    camera_matrix = np.array(camera_matrix, dtype=np.float32).reshape(4, 4)
    dscene.camera.set_transform(camera_matrix)
    dscene.params["PerspectiveCamera.to_world"] = camera_matrix
    dscene.params.update()

class FunctionWrap:
    def __init__(self, scene: mi.Scene, ui, set_camera: bool = False, rfilter_idx: int = None):
        self.ui = ui
        scene = scene
        size = scene.sensors()[0].film().size()
        self.size_ori = [size[0], size[1]]
        self.scene_params = mi.traverse(scene)
        self.film_size_str = ""
        for k in self.scene_params.keys():
            if k.endswith("film.size"):
                self.film_size_str = k
                break
        self.scale = 1.0
        self.render_size = [size[0], size[1]]

        # ref image & calculate error
        self.img_ref: torch.Tensor = None
        self.show_ref: bool = False
        self.show_ref: bool = False
        self.should_calc_error: bool = False
        self.ERROR_TYPES = ["MSE", "relMSE", "MAPE", "MAE", "SMAPE"]
        self.error_type_idx: int = 2  # MAPE

        # rfilter
        self.film: mi.Film = scene.sensors()[0].film()
        self.FILTER_TYPES = ["box", "tent", "lanczos", "mitchell", "catmullrom", "gaussian"]
        if(rfilter_idx is not None):
            self.rfilter_idx = rfilter_idx
            self.film.set_rfilter(self.FILTER_TYPES[self.rfilter_idx])
        else:
            self.rfilter_idx = self.get_rfilter_idx(self.film.rfilter())

        if set_camera:
            self.set_camera_matrix()

    def get_rfilter_idx(self, rfilter: mi.ReconstructionFilter) -> int:
        rfilter = str(rfilter)
        filters = ["BoxFilter", "TentFilter", "LanczosSincFilter",
                   "MitchellNetravaliFilter", "CatmullRomFilter", "Gaussian"]
        for i, f in enumerate(filters):
            if rfilter.startswith(f):
                return i

    def render_ui_before(self) -> bool:
        value_changed = False
        if imgui.tree_node("Film Options", imgui.NONE):
            vc, self.rfilter_idx = imgui.combo("Filter", self.rfilter_idx, self.FILTER_TYPES, len(self.FILTER_TYPES))
            if vc:
                self.film.set_rfilter(self.FILTER_TYPES[self.rfilter_idx])
                value_changed = True

            set2x = imgui.button("Set 2x")
            if set2x:
                self.scale = 0.5

            vc, self.scale = imgui.slider_float("Scale", self.scale, 0.1, 1.0)
            if vc or set2x:
                self.render_size = [int(self.scale * i) for i in self.size_ori]
                self.scene_params[self.film_size_str] = self.render_size
                self.scene_params.update()
                self.ui.check_and_update_texture_size(*self.render_size)
                value_changed = True

            imgui.text("Original Size: {} x {}".format(*self.size_ori))
            imgui.text("Render Size: {} x {}".format(*self.render_size))

            imgui.tree_pop()

        return value_changed

    def render_ui_after(self, img: torch.Tensor) -> torch.Tensor:
        return img

    def print_camera_matrix(self):
        camera_matrix = self.dscene.camera.get_transform()
        print("Matrix:\n{}".format(camera_matrix))
        camera_matrix = camera_matrix.reshape(-1).tolist()
        camera_matrix = [str(i) for i in camera_matrix]
        camera_matrix_str = ",".join(camera_matrix)
        print("Matrix(for camera.txt):\n{}".format(camera_matrix_str))

    def set_camera_matrix(self):
        camera_txt = os.path.join(CURRENT_DIR, "camera.txt")
        if not os.path.exists(camera_txt):
            print("\033[0;31m[\{}]\033[0m not exists".format(camera_txt))
            return

        load_camera_txt_and_set(self.dscene, camera_txt)

    def load_refexr(self, path: str):
        self.img_ref = mi.TensorXf(mi.Bitmap(path)).torch().cuda()

    def calc_error(self, update_frame: bool, img: torch.Tensor) -> Tuple[bool, bool, torch.Tensor]:
        t_show_ref = False
        if (self.img_ref is not None):
            if imgui.tree_node("Error", imgui.TREE_NODE_NONE):
                vc, self.show_ref = imgui.checkbox("Show Reference", self.show_ref)
                update_frame = update_frame or vc
                if self.show_ref:
                    img = self.img_ref
                    t_show_ref = True
                    update_frame = False
                if (not self.show_ref):
                    _, self.error_type_idx = imgui.combo("Error Type", self.error_type_idx, self.ERROR_TYPES, len(self.ERROR_TYPES))
                    self.should_calc_error = imgui.button("Calculate Error")
                imgui.tree_pop()

        return t_show_ref, update_frame, img

    def get_should_calc_error(self):
        return self.should_calc_error

    def calc_error_run(self, img):
        '''
        make sure you call this function after after get True from get_should_calc_error()
        '''
        error_type = self.ERROR_TYPES[self.error_type_idx]
        e1 = compute_metric_torch(img, self.img_ref, error_type)
        # e2 = compute_metric(img.cpu().numpy(), self.img_ref.cpu().numpy(), error_type)
        error_str = "{} Error: {:.6f}".format(error_type, e1)
        min_max = [torch.flatten(img).min(), torch.flatten(img).max()]
        min_max_ref = [torch.flatten(self.img_ref).min(), torch.flatten(self.img_ref).max()]
        print("{:s}, out: ({:.6f}, {:.6f}), ref: ({:.6f}, {:.6f})".format(
            error_str, min_max[0], min_max[1], min_max_ref[0], min_max_ref[1]))
