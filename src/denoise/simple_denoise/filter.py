import os
import sys

from OpenGL.GL import *
from OpenGL.GL.ARB.pixel_buffer_object import *
import imgui
import numpy as np
from enum import Enum
from cuda.bindings import runtime as cudart
import mitsuba as mi
from integrator.mi_albedo_normal import BAlbedoNormalDepthIntegrator
import time
import torch

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(CURRENT_DIR, "../"))
sys.path.append(os.path.join(CURRENT_DIR, "../../"))
from denoise.ogl.gl_helper import OpenGLHelper as glh
from denoise.ogl.compute_task import ComputeTask
from src.viewer.ui import UI, check_cuda_error
from simple_denoise.pynis import NIS


class KernelType(Enum):
    NONE = 0
    AVERAGE = 1
    GAUSSIAN = 2
    MEDIAN = 3
    NIS = 4
    BILATERAL = 5
    Depth = 6
    CUDA_AA = 7
    FXAA = 8


class FilterTasks(ComputeTask):

    def __init__(self, ui: UI, scene: mi.Scene, width: int = None, height: int = None, texture_size=None):
        '''
        If ui is not None, then width/height/texture_size won't be used.
        If ui is None(headless mode), then width/height/texture_size must be set, ui_use_gpu will be set True.
        In headless mode, we assume width/height/texture_size won't change.
        '''

        self.name = "Simple Denoise"
        super().__init__(self.name, "denoise/color.comp")

        if ui is not None:
            self.ui_use_gpu = ui.gpu
        else:
            assert (width is not None) and (height is not None) and (texture_size is not None)
            self.ui_use_gpu = True
        self.scene: mi.Scene = scene
        self.sensor: mi.Sensor = scene.sensors()[0]

        self.TYPES = [str(i) for i in KernelType.__members__]
        # self.TYPES = [(i[0].upper() + i[1:].lower()) for i in self.TYPES]
        self.need_kernel_size: list = None
        self.need_sigma: list = None
        self.create_configs()

        self.kernel_type: KernelType = KernelType.FXAA  # TODO: set default to NONE
        self.kernel_size = 3
        self.sigma = 1.0
        self.value_sigma = 1.0
        self.LOCAL_SIZE = 16

        # NIS
        self.nis_task: NIS = None
        self.output_tex_size_same_with_window = False

        # TODO: tooooo ugly, should be managed by TextureManager
        if (ui is not None):
            self.window_size = [ui.width, ui.height]
            self.texture_size = ui.texture_size
            self.physical_texture_size = ui.texture_size  # real texture size
            self.output_texture = self.create_texture(*ui.texture_size)
        else:
            self.window_size = [width, height]
            self.texture_size = texture_size
            self.physical_texture_size = texture_size
            self.output_texture = self.create_texture(*texture_size)

        self.depth_texture = None
        self.depth_texture_pbo = None
        self.depth_texture_pbo_buf = None

        # cuda_aa
        self.albedo_depth_normal_texture = []
        self.albedo_depth_normal_texture_pbo = []
        self.albedo_depth_normal_texture_pbo_buf = []
        self.aux_integrator: mi.Integrator = None
        self.cuda_aa_spp = 1
        self.cuda_aa_depth_threshold = 0.01
        self.cuda_aa_albedo_threshold = 0.01
        self.cuda_aa_normal_threshold = 0.01

        # headless
        self.headless_tex = None
        self.headless_tex_pbo = None
        self.headless_tex_pbo_buf = None
        self.tex_out_pbo = None
        self.tex_out_pbo_buf = None

    def create_configs(self):
        # need_kernel_size, need_sigma
        self.need_kernel_size = [False] * len(self.TYPES)
        self.need_sigma = [False] * len(self.TYPES)

        ktv = KernelType.AVERAGE.value
        self.need_kernel_size[ktv] = True
        ktv = KernelType.GAUSSIAN.value
        self.need_sigma[ktv] = True
        ktv = KernelType.MEDIAN.value
        self.need_kernel_size[ktv] = True
        # ktv = KernelType.NIS.value
        ktv = KernelType.BILATERAL.value
        self.need_sigma[ktv] = True
        # ktv = KernelType.Depth.value
        ktv = KernelType.CUDA_AA.value
        self.need_kernel_size[ktv] = True

    def get_name(self):
        return self.name

    def create_simple_texture(self, width, height, internal_format):
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        if internal_format == GL_R32F:
            glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width, height, 0, GL_RED, GL_FLOAT, None)
        elif internal_format == GL_RGBA32F:
            glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width, height, 0, GL_RGBA, GL_FLOAT, None)
        elif internal_format == GL_RGB32F:
            glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width, height, 0, GL_RGB, GL_FLOAT, None)
        else:
            raise Exception("Not Implemented")

        glBindTexture(GL_TEXTURE_2D, 0)

        return texture

    def create_texture(self, width, height):
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 1)

        # data_test = np.ones((width, height, 4), dtype=np.float32) * 0.5
        # glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, data_test)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, None)
        glGenerateMipmap(GL_TEXTURE_2D)

        glBindTexture(GL_TEXTURE_2D, 0)

        return texture

    def run(self, **kwargs):
        if self.kernel_type == KernelType.NIS:
            if not self.nis_task.is_bilinear():
                self.run_nis(**kwargs)
                return
            else:
                pass
        self.run_normal(**kwargs)

    def run_normal(self, **kwargs):
        group_size = kwargs.get('group_size')
        assert group_size is not None
        group_size = list(group_size)
        assert len(group_size) == 2

        group_size[0] = np.ceil(group_size[0] / self.LOCAL_SIZE)
        group_size[1] = np.ceil(group_size[1] / self.LOCAL_SIZE)
        group_size.append(1)
        group_size = [int(x) for x in group_size]

        glUseProgram(self.program)
        tex_in = kwargs.get('tex_input')
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex_in)
        glBindImageTexture(0, tex_in, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F)  # binding = 0
        glGenerateMipmap(GL_TEXTURE_2D)
        # glUniform1i(glGetUniformLocation(self.program, "img_input"), 0) # we have already bind it in the shader
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self.output_texture)
        glBindImageTexture(1, self.output_texture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F)  # binding = 1
        if self.need_depth():
            glActiveTexture(GL_TEXTURE2)
            glBindTexture(GL_TEXTURE_2D, self.depth_texture)
            glBindImageTexture(2, self.depth_texture, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F)  # binding = 2
        if self.is_cuda_aa():
            glActiveTexture(GL_TEXTURE3)
            glBindTexture(GL_TEXTURE_2D, self.albedo_depth_normal_texture[0])
            glBindImageTexture(3, self.albedo_depth_normal_texture[0], 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F)
            glActiveTexture(GL_TEXTURE4)
            glBindTexture(GL_TEXTURE_2D, self.albedo_depth_normal_texture[1])
            glBindImageTexture(4, self.albedo_depth_normal_texture[1], 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F)
            glActiveTexture(GL_TEXTURE5)
            glBindTexture(GL_TEXTURE_2D, self.albedo_depth_normal_texture[2])
            glBindImageTexture(5, self.albedo_depth_normal_texture[2], 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F)

        ktv = self.kernel_type.value
        if (self.kernel_type == KernelType.NIS) and (self.nis_task is not None) and (self.nis_task.is_bilinear()):
            ktv = KernelType.NONE.value
        v1x = self.kernel_size
        v1y = ktv
        v2x = self.sigma
        v2y = self.value_sigma
        v2z = 0.0
        if (self.is_cuda_aa()):
            v2x = self.cuda_aa_depth_threshold
            v2y = self.cuda_aa_albedo_threshold
            v2z = self.cuda_aa_normal_threshold
        glUniform4i(glGetUniformLocation(self.program, "v1"), v1x, v1y, 0, 0)
        glUniform4f(glGetUniformLocation(self.program, "v2"), v2x, v2y, v2z, 0.0)
        glDispatchCompute(*group_size)
        # make sure writing to image has finished before read
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)
        glBindTexture(GL_TEXTURE_2D, 0)
        glUseProgram(0)

    def run_nis(self, **kwargs):
        program = self.nis_task.get_program()
        glUseProgram(program)

        self.nis_task.update(self.texture_size, self.window_size)

        # attach constant buffer(binding = 0)
        self.nis_task.bind_cbuffer(self.exposure)

        binding = -1  # texture binding points
        # attach images
        tex = self.output_texture
        binding += 1
        location = glGetUniformLocation(program, "out_texture")
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex)
        glBindImageTexture(binding, tex, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F)
        glUniform1i(location, binding)

        tex = kwargs.get('tex_input')
        binding += 1
        location = glGetUniformLocation(program, "SPIRV_Cross_Combinedin_texturesamplerLinearClamp")
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glBindImageTexture(binding, tex, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F)
        glUniform1i(location, binding)

        # packed params
        tex = self.nis_task.coef_scale_fp16_texture
        binding += 1
        location = glGetUniformLocation(program, "SPIRV_Cross_Combinedcoef_scalersamplerLinearClamp")
        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, tex)
        glBindImageTexture(binding, tex, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA16F)
        glUniform1i(location, binding)

        tex = self.nis_task.coef_usm_fp16_texture
        binding += 1
        location = glGetUniformLocation(program, "SPIRV_Cross_Combinedcoef_usmsamplerLinearClamp")
        glActiveTexture(GL_TEXTURE3)
        glBindTexture(GL_TEXTURE_2D, tex)
        glBindImageTexture(binding, tex, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA16F)
        glUniform1i(location, binding)

        # run
        w, h = self.nis_task.get_optimal_dispatch_size()
        glDispatchCompute(w, h, 1)
        # make sure writing to image has finished before read
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

        # exit
        glBindTexture(GL_TEXTURE_2D, 0)
        glUseProgram(0)

    def set(self, **kwargs):
        pass
        # example:
        # self.use_tonemapping = kwargs.get('use_tonemapping', True)

    def render_ui(self, integrator: mi.Integrator):
        # if self.kernel_type != KernelType.NIS or not self.nis_task.is_NV_scaler():
        self.output_tex_size_same_with_window = False
        value_changed = False
        ktv: int = self.kernel_type.value
        vc, ktv = imgui.combo("kernel type", ktv, self.TYPES)
        self.kernel_type = KernelType(ktv)

        if(ktv in [KernelType.NIS.value, KernelType.CUDA_AA.value]):
            ktv = 0
            self.kernel_type = KernelType(ktv)

        value_changed = value_changed or vc

        if self.need_kernel_size[ktv]:
            vc, self.kernel_size = imgui.slider_int("kernel size", self.kernel_size, 1, 5)
            ks = self.kernel_size * 2 - 1
            imgui.text_ansi("kernel size: {} x {}".format(ks, ks))
            value_changed = value_changed or vc
            if self.kernel_type == KernelType.MEDIAN:
                imgui.text_ansi("Median Filter is Slow! Bubble Sort! Max Kernel Size = 9")

        if self.need_sigma[ktv]:
            vc, self.sigma = imgui.slider_float("sigma", self.sigma, 0.1, 5.0)
            imgui.text_ansi("2 sigma: kernel size = {}".format(int(np.ceil(self.sigma * 2))))
            value_changed = value_changed or vc
        if self.kernel_type == KernelType.BILATERAL:
            vc, self.value_sigma = imgui.slider_float("value sigma", self.value_sigma, 0.1, 5.0)
            imgui.text_ansi("2 sigma: kernel size = {}".format(int(np.ceil(self.value_sigma * 2))))
            imgui.text_ansi("final kernel size = {}".format(int(np.ceil(max(self.sigma, self.value_sigma) * 2))))
            value_changed = value_changed or vc
        elif self.kernel_type == KernelType.NIS:
            if self.nis_task is None:
                self.nis_task = NIS(self.window_size, self.texture_size)
            self.nis_task.render_ui()
            if (self.nis_task.is_NV_scaler()):
                self.output_tex_size_same_with_window = True
        self.resize(*self.texture_size)

        if self.need_depth():
            integrator = mi.load_dict({'type': 'b_albedo_depth'})

        if self.is_cuda_aa():
            vc, self.cuda_aa_spp = imgui.slider_int("aux spp", self.cuda_aa_spp, 1, 64)
            value_changed = value_changed or vc
            vc, self.cuda_aa_depth_threshold = imgui.slider_float("depth threshold", self.cuda_aa_depth_threshold, 0.001, 1.0)
            value_changed = value_changed or vc
            vc, self.cuda_aa_albedo_threshold = imgui.slider_float("albedo threshold", self.cuda_aa_albedo_threshold, 0.001, 1.0)
            value_changed = value_changed or vc
            vc, self.cuda_aa_normal_threshold = imgui.slider_float("normal threshold", self.cuda_aa_normal_threshold, 0.001, 1.0)
            value_changed = value_changed or vc

        return value_changed, integrator

    def resize(self, width, height):
        self.texture_size = [width, height]

        if self.kernel_type == KernelType.NIS:
            self.nis_task.resize(width, height)

        # check should be update
        new_size = None
        if (self.output_tex_size_same_with_window):
            new_size = self.window_size
        else:
            new_size = self.texture_size

        if (new_size == self.physical_texture_size):
            return

        self.physical_texture_size = new_size

        glDeleteTextures([self.output_texture])
        self.output_texture = self.create_texture(*new_size)
        if self.depth_texture is not None:
            glDeleteTextures([self.depth_texture])
            self.create_depth_texture()

        if self.albedo_depth_normal_texture != []:
            glDeleteTextures(self.albedo_depth_normal_texture)
            self.albedo_depth_normal_texture = []
            self.create_albedo_depth_normal_texture()

    def need_depth(self):
        return self.kernel_type == KernelType.Depth
        # return self.kernel_type == KernelType.Bilateral

    def is_cuda_aa(self):
        return self.kernel_type == KernelType.CUDA_AA

    def create_pbo_and_buf(self, width, height, internal_format):
        size = -1
        if internal_format == GL_R32F:
            size = width * height * 4
        elif internal_format == GL_RGBA32F:
            size = width * height * 16
        elif internal_format == GL_RGB32F:
            size = width * height * 12
        else:
            raise Exception("Not Implemented")
        pbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, pbo)
        glBufferData(GL_ARRAY_BUFFER, size, None, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        cres, bufobj = cudart.cudaGraphicsGLRegisterBuffer(pbo, cudart.cudaGraphicsRegisterFlags(0))
        check_cuda_error(cres)

        return pbo, bufobj

    def create_depth_texture(self):
        self.depth_texture = self.create_simple_texture(*self.physical_texture_size, GL_R32F)
        if (self.ui_use_gpu):
            if self.depth_texture_pbo != None:
                cres, = cudart.cudaGraphicsUnregisterResource(self.depth_texture_pbo_buf)
                check_cuda_error(cres)
                glDeleteBuffers(1, [self.depth_texture_pbo])
            self.depth_texture_pbo, self.depth_texture_pbo_buf = self.create_pbo_and_buf(*self.physical_texture_size, GL_R32F)

    def create_albedo_depth_normal_texture(self):
        # albedo, depth, normal
        should_del = self.albedo_depth_normal_texture_pbo != []
        pbos = self.albedo_depth_normal_texture_pbo
        pbo_bufs = self.albedo_depth_normal_texture_pbo_buf

        self.albedo_depth_normal_texture_pbo = []
        self.albedo_depth_normal_texture_pbo_buf = []

        formats = [GL_RGBA32F, GL_R32F, GL_RGBA32F]
        for i in range(3):
            texture = self.create_simple_texture(*self.physical_texture_size, formats[i])
            self.albedo_depth_normal_texture.append(texture)
            if (self.ui_use_gpu):
                if should_del:
                    cres, = cudart.cudaGraphicsUnregisterResource(pbo_bufs[i])
                    check_cuda_error(cres)
                    glDeleteBuffers(1, pbos[i])
                pbo, pbo_buf = self.create_pbo_and_buf(*self.physical_texture_size, formats[i])
                self.albedo_depth_normal_texture_pbo.append(pbo)
                self.albedo_depth_normal_texture_pbo_buf.append(pbo_buf)

    def record_depth(self, depth, should_normalize=True):
        if (should_normalize):
            min_depth, max_depth = depth.min(), depth.max()
            depth = (depth - min_depth) / (max_depth - min_depth)

        if (self.depth_texture is None):
            self.create_depth_texture()

        if (self.ui_use_gpu):
            # cuda -> pbo
            cres, = cudart.cudaGraphicsMapResources(1, self.depth_texture_pbo_buf, 0)
            check_cuda_error(cres)
            cres, ptr, size = cudart.cudaGraphicsResourceGetMappedPointer(self.depth_texture_pbo_buf)
            check_cuda_error(cres)
            cres, = cudart.cudaMemcpy(ptr, depth.data_ptr(), size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)
            check_cuda_error(cres)
            cres, = cudart.cudaGraphicsUnmapResources(1, self.depth_texture_pbo_buf, 0)
            check_cuda_error(cres)
            # pbo -> texture
            glBindTexture(GL_TEXTURE_2D, self.depth_texture)
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self.depth_texture_pbo)
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, *self.physical_texture_size, GL_RED, GL_FLOAT, None)
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
            glBindTexture(GL_TEXTURE_2D, 0)
        else:
            glBindTexture(GL_TEXTURE_2D, self.depth_texture)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, *self.texture_size, 0, GL_RED, GL_FLOAT, depth.cpu().numpy())
            glBindTexture(GL_TEXTURE_2D, 0)

    def record_albedo_depth_normal(self, albedo, depth, normal, depth_should_normalize=True):
        if (depth_should_normalize):
            min_depth, max_depth = depth.min(), depth.max()
            depth = (depth - min_depth) / (max_depth - min_depth)

        if (self.albedo_depth_normal_texture == []):
            self.create_albedo_depth_normal_texture()

        datas = [albedo, depth, normal]
        bufs = self.albedo_depth_normal_texture_pbo_buf
        pbos = self.albedo_depth_normal_texture_pbo
        textures = self.albedo_depth_normal_texture
        formats = [GL_RGBA32F, GL_R32F, GL_RGBA32F]
        rgb_formats = [GL_RGB, GL_RED, GL_RGB]

        for i in range(3):
            if (self.ui_use_gpu):
                # cuda -> pbo
                cres, = cudart.cudaGraphicsMapResources(1, bufs[i], 0)
                check_cuda_error(cres)
                cres, ptr, size = cudart.cudaGraphicsResourceGetMappedPointer(bufs[i])
                check_cuda_error(cres)
                cres, = cudart.cudaMemcpy(ptr, datas[i].data_ptr(), size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)
                check_cuda_error(cres)
                cres, = cudart.cudaGraphicsUnmapResources(1, bufs[i], 0)
                check_cuda_error(cres)
                # pbo -> texture
                glBindTexture(GL_TEXTURE_2D, textures[i])
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbos[i])
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, *self.physical_texture_size, rgb_formats[i], GL_FLOAT, None)
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
                glBindTexture(GL_TEXTURE_2D, 0)
            else:
                glBindTexture(GL_TEXTURE_2D, self.depth_texture)
                glTexImage2D(GL_TEXTURE_2D, 0, formats[i], *self.texture_size, 0, rgb_formats[i], GL_FLOAT, datas[i].cpu().numpy())
                glBindTexture(GL_TEXTURE_2D, 0)

    def release(self):
        glDeleteTextures([self.output_texture])
        if self.nis_task is not None:
            self.nis_task.release()
        super().release()

    def denoise(self, img):
        img = img.torch()
        if self.need_depth():
            depth = img[::, ::, -1].clone()
            self.record_depth(depth)
        if self.is_cuda_aa():
            if (self.aux_integrator is None):
                self.aux_integrator = mi.load_dict({'type': 'b_albedo_normal_depth'})
            # TODO: [opti] update it when scene changed
            aux_tensor: torch.Tensor = mi.render(self.scene, integrator=self.aux_integrator,
                                                 sensor=self.sensor, spp=self.cuda_aa_spp, seed=time.time_ns() % (4194304)).torch()
            albedo = aux_tensor[::, ::, 0:3].clone()
            normal = aux_tensor[::, ::, 3:6].clone()
            depth = aux_tensor[::, ::, -1].clone()
            self.record_albedo_depth_normal(albedo, depth, normal)
        return img[::, ::, :3].clone()

    def fetch_denoised_result_headless(self, img: torch.Tensor, copy_back: bool) -> torch.Tensor:
        '''
        Make sure use it in headless mode(no UI).
        Here, we resume the size of the texture won't change and the texture is [cuda<->opengl] interop.
        '''
        assert self.ui_use_gpu

        # [1] guard
        if (self.headless_tex is None):
            self.headless_tex = self.create_simple_texture(*self.texture_size, GL_RGB32F)
            self.headless_tex_pbo, self.headless_tex_pbo_buf = self.create_pbo_and_buf(*self.texture_size, GL_RGB32F)
            self.tex_out_pbo, self.tex_out_pbo_buf = self.create_pbo_and_buf(*self.texture_size, GL_RGB32F)
            self.a = torch.zeros_like(img).cuda()

        # [2] copy data(GPU mode)
        cres, = cudart.cudaGraphicsMapResources(1, self.headless_tex_pbo_buf, 0)
        check_cuda_error(cres)
        cres, ptr, size = cudart.cudaGraphicsResourceGetMappedPointer(self.headless_tex_pbo_buf)
        check_cuda_error(cres)
        cres, = cudart.cudaMemcpy(ptr, img.data_ptr(), size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)
        check_cuda_error(cres)
        cres, = cudart.cudaGraphicsUnmapResources(1, self.headless_tex_pbo_buf, 0)
        check_cuda_error(cres)
        # pbo -> texture
        glBindTexture(GL_TEXTURE_2D, self.headless_tex)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self.headless_tex_pbo)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, *self.physical_texture_size, GL_RGB, GL_FLOAT, None)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
        glBindTexture(GL_TEXTURE_2D, 0)

        # [3] run
        self.run(tex_input=self.headless_tex, group_size=self.texture_size)

        img_ret = None
        if (copy_back):
            # [4] copy back
            # slow as it copy data from GPU to CPU
            glBindTexture(GL_TEXTURE_2D, self.output_texture)
            frame_buffer = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_FLOAT, None)
            glBindTexture(GL_TEXTURE_2D, 0)
            img_ret = torch.tensor(frame_buffer, dtype=torch.float32).reshape(self.window_size[1], self.window_size[0], 3)

        return img_ret
