import os
import sys
import imgui
from enum import Enum
from OpenGL.GL import *
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(CURRENT_DIR, "../"))

from src.denoise.ogl.gl_helper import OpenGLHelper as glh
# from src.denoise.simple_denoise.pyNISConfigWrapper import pyNISConfig


class NISType(Enum):
    NV_SCALER = 0
    NV_SHARPEN = 1
    Bilinear = 2


class NIS(object):
    def __init__(self, size, texture_size):
        self.name = "NIS"

        self.shapern_program, self.scaler_program = self.create_programs()
        self.nis_params: pyNISConfig = pyNISConfig()
        self.nis_params_cbuf = self.create_nis_params_cbuf()

        self.filter_type: NISType = NISType.NV_SCALER
        self.sharpness = 0.0
        self.enable_nv_scaler = False
        self.should_update_config = False

        self.coef_scale_fp16_texture = None
        self.coef_usm_fp16_texture = None
        self.create_param_textures()

        self.input_size = [texture_size[0], texture_size[1]]
        self.output_size = [size[0], size[1]]

    def texture_simple(self):
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        return texture

    def create_param_textures(self):
        kFilterSize, kPhaseCount = self.nis_params.get_shader_params()
        print("kFilterSize = {}, kPhaseCount = {}".format(kFilterSize, kPhaseCount))

        # coef_scale_fp16
        tex = self.texture_simple()
        self.coef_scale_fp16_texture = tex
        data: np.ndarray = self.nis_params.get_coef_scale_fp16()
        data = data.view(dtype=np.float16)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, kFilterSize / 4, kPhaseCount, 0, GL_RGBA, GL_HALF_FLOAT, data)
        glBindTexture(GL_TEXTURE_2D, 0)

        # coef_usm_fp16
        tex = self.texture_simple()
        self.coef_usm_fp16_texture = tex
        data = self.nis_params.get_coef_usm_fp16()
        data = data.view(dtype=np.float16)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, kFilterSize / 4, kPhaseCount, 0, GL_RGBA, GL_HALF_FLOAT, data)
        glBindTexture(GL_TEXTURE_2D, 0)

    def get_optimal_dispatch_size(self):
        # OptimalBlockWidth, OptimalBlockHeight, OptimalThreadGroupSize
        w, h, g = self.nis_params.get_optimal_dispatch_size()
        ref_size = None
        if self.filter_type == NISType.NV_SHARPEN:
            ref_size = self.input_size
        elif self.filter_type == NISType.NV_SCALER:
            ref_size = self.output_size
        else:
            print("\033[31m[ERROR]\033[0m Won't reach here")

        w = int(np.ceil(ref_size[0] / w))
        h = int(np.ceil(ref_size[1] / h))
        return w, h

    def bind_cbuffer(self, reserved0: float):
        reserved0_pos = 26
        buf_size = self.nis_params.get_size()
        glBindBuffer(GL_UNIFORM_BUFFER, self.nis_params_cbuf)
        data = self.nis_params.get_config()
        # print(data.view(np.int32)[18:26])
        data.view(np.float32)[reserved0_pos] = reserved0
        glBufferSubData(GL_UNIFORM_BUFFER, 0, buf_size, data)
        # glBindBufferRange(GL_UNIFORM_BUFFER, binding, self.nis_params_cbuf, 0, buf_size)
        glBindBuffer(GL_UNIFORM_BUFFER, 0)

    # must be called before dispatch
    def update(self, input_size, output_size):
        should_update = False
        if self.input_size != input_size:
            self.input_size = input_size
            should_update = True
        if self.output_size != output_size:
            self.output_size = output_size
            should_update = True
        should_update = should_update or self.should_update_config
        self.should_update_config = False
        if should_update:
            if self.filter_type == NISType.NV_SCALER:
                self.nis_params.NVScalerUpdateConfig(self.sharpness, *self.input_size, *self.output_size)
            elif self.filter_type == NISType.NV_SHARPEN:
                self.nis_params.NVSharpenUpdateConfig(self.sharpness, *self.input_size)
            else:
                print("\033[31mERROR: invalid filter type\033[0m")

    def create_nis_params_cbuf(self):
        binding = 0
        ubo = glGetUniformBlockIndex(self.scaler_program, "const_buffer")
        glUniformBlockBinding(self.scaler_program, ubo, binding)
        ubo = glGetUniformBlockIndex(self.shapern_program, "const_buffer")
        glUniformBlockBinding(self.shapern_program, ubo, binding)

        buf_size = self.nis_params.get_size()
        cbuf = glGenBuffers(1)
        glBindBuffer(GL_UNIFORM_BUFFER, cbuf)
        glBufferData(GL_UNIFORM_BUFFER, buf_size, None, GL_DYNAMIC_DRAW)
        glBindBufferRange(GL_UNIFORM_BUFFER, binding, cbuf, 0, buf_size)
        glBindBuffer(GL_UNIFORM_BUFFER, 0)
        return cbuf

    def create_programs(self):
        sharpen_program = None
        scaler_program = None

        sharpen_program = glh.create_compute_program("NIS/generated/nis_sharpen.comp")
        scaler_program = glh.create_compute_program("NIS/generated/nis_scaler.comp")

        return sharpen_program, scaler_program

    def render_ui(self):
        value_changed = False
        vc, ftv = imgui.combo("filter type", self.filter_type.value, ["NVScaler", "NVSharpen", "Bilinear"])
        if vc:
            self.filter_type = NISType(ftv)
        value_changed = value_changed or vc

        if self.filter_type in [NISType.NV_SCALER, NISType.NV_SHARPEN]:
            self.enable_nv_scaler = True
            vc, self.sharpness = imgui.slider_float(R"Sharpness(0% - 100%)", self.sharpness, 0.0, 1.0, R"%.3f")
            self.should_update_config = self.should_update_config or vc
            value_changed = value_changed or vc
        else:
            self.enable_nv_scaler = False

    def get_program(self):
        if self.filter_type == NISType.NV_SCALER or self.filter_type == NISType.Bilinear:
            return self.scaler_program
        elif self.filter_type == NISType.NV_SHARPEN:
            return self.shapern_program
        else:
            return None

    def is_NV_scaler(self):
        return self.filter_type == NISType.NV_SCALER

    def is_bilinear(self):
        return self.filter_type == NISType.Bilinear

    def resize(self, width, height):
        self.input_size = [width, height]
        self.should_update_config = True

    def release(self):
        glDeleteBuffers(1, [self.nis_params_cbuf])
        glDeleteTextures(1, [self.coef_scale_fp16_texture])
        glDeleteTextures(1, [self.coef_usm_fp16_texture])
        glDeleteProgram(self.shapern_program)
        glDeleteProgram(self.scaler_program)
