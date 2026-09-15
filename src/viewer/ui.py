import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
from OpenGL.GL.ARB.pixel_buffer_object import *
import imgui
from imgui.integrations.glfw import GlfwRenderer
import drjit as dr
import mitsuba as mi

from cuda.bindings import runtime as cudart

import numpy as np
import drjit as dr
import torch
import time
import os
import warnings


def check_cuda_error(cres: cudart.cudaError_t):
    if cres != cudart.cudaError_t.cudaSuccess:
        print(str(cres))
        print("\033[31mCUDA error: {}\033[0m".format(cres))


class UI:

    def __init__(self, width, height, camera, name="Render", bbox=None):
        self.gpu = True  # TODO
        self.width = width
        self.height = height
        self.name = name

        self.camera = camera
        self.first_mouse = True
        self.prev_x = 0
        self.prev_y = 0

        self.current = time.time()
        self.duration = 0
        self.frames = 0
        self.frame_rate = 1.0

        self.use_tonemapping = True
        self.exposure = 1.0

        self.speed = 1.0
        if bbox is not None:
            scale = bbox.max - bbox.min
            self.scale = max(scale[0], scale[1], scale[2])
        else:
            self.scale = 1.0

        # initialize glfw
        if not glfw.init():
            print("Failed to initialize GLFW")
            exit()

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 6)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        window = glfw.create_window(width, height, name, None, None)

        if not window:
            print("Failed to create window")
            glfw.terminate()
            exit()

        glfw.make_context_current(window)

        glViewport(0, 0, width, height)

        glfw.swap_interval(0)
        self.window = window

        # initialize imgui
        imgui.create_context()
        self.impl = GlfwRenderer(window)

        # create shader, vao, texture
        file_path = os.path.dirname(os.path.realpath(__file__))
        self.program = self.create_program(
            file_path + "/shader/hello.vert",
            file_path + "/shader/hello.frag"
        )
        self.vao = self.create_vao()

        self.texture_size = (-1, -1)
        self.texture = None
        self.pbo = None
        self.bufobj = None
        self.compute_task = None

        self.check_and_update_texture_size(width, height)
    
    def set_vars(self, scene, camera):
        self.scene = scene
        self.camera = camera
        self.width = camera.width
        self.height = camera.height
        bbox = scene.bbox()
        scale = bbox.max - bbox.min
        self.scale = max(scale[0], scale[1], scale[2])

        # glfw.set_window_size(self.window, self.width, self.height)
        glViewport(0, 0, self.width, self.height)
        self.check_and_update_texture_size(self.width, self.height)

    def set_tonemapping(self, use_tonemapping: bool, exposure: float = None):
        self.use_tonemapping = use_tonemapping
        if exposure != None:
            self.exposure = exposure

    def close(self):
        if self.compute_task != None:
            self.compute_task.release()
        if self.gpu:
            cres, = cudart.cudaGraphicsUnregisterResource(self.bufobj)
            check_cuda_error(cres)
            glDeleteBuffers(1, [self.pbo])

        glDeleteTextures([self.texture])
        glDeleteVertexArrays(1, [self.vao])
        glDeleteProgram(self.program)

        self.impl.shutdown()
        glfw.destroy_window(self.window)
        glfw.terminate()

    def create_program(self, vertex_path, fragment_path):

        with open(vertex_path, "r") as f:
            vertex_shader = f.read()

        with open(fragment_path, "r") as f:
            fragment_shader = f.read()

        vertex = OpenGL.GL.shaders.compileShader(
            vertex_shader, GL_VERTEX_SHADER)
        fragment = OpenGL.GL.shaders.compileShader(
            fragment_shader, GL_FRAGMENT_SHADER)

        return OpenGL.GL.shaders.compileProgram(vertex, fragment)

    def create_vao(self):

        # flip vertically
        quad = np.array([
            # position 2, texcoord 2
            -1.0, 1.0, 0.0, 0.0,
            -1.0, -1.0, 0.0, 1.0,
            1.0, -1.0, 1.0, 1.0,

            -1.0, 1.0, 0.0, 0.0,
            1.0, -1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 0.0
        ], dtype=np.float32)

        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)
        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, quad.nbytes, quad, GL_STATIC_DRAW)

        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 *
                              quad.itemsize, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 *
                              quad.itemsize, ctypes.c_void_p(2 * quad.itemsize))
        glEnableVertexAttribArray(1)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
        glDeleteBuffers(1, [vbo])

        return vao

    def create_pbo(self, w, h):

        data = np.zeros((w * h * 3), dtype=np.float32)
        pbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, pbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        return pbo

    def create_texture(self, width, height):

        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 1)
        # glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, None)
        glTexStorage2D(GL_TEXTURE_2D, 2, GL_RGBA32F, width, height)  # mipmap levels = 1
        # glGenerateMipmap(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, 0)

        return texture

    def process_input(self):
        if glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(self.window, True)

        move_speed = self.frame_rate * self.speed * self.scale
        rotate_speed = self.frame_rate * self.speed * 10

        # Camera controls
        if glfw.get_key(self.window, glfw.KEY_W) == glfw.PRESS:
            self.camera.move(0, move_speed)
        if glfw.get_key(self.window, glfw.KEY_S) == glfw.PRESS:
            self.camera.move(0, -move_speed)
        if glfw.get_key(self.window, glfw.KEY_A) == glfw.PRESS:
            self.camera.move(2, move_speed)
        if glfw.get_key(self.window, glfw.KEY_D) == glfw.PRESS:
            self.camera.move(2, -move_speed)
        if glfw.get_key(self.window, glfw.KEY_SPACE) == glfw.PRESS:
            self.camera.move(1, move_speed)
        if glfw.get_key(self.window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS:
            self.camera.move(1, -move_speed)

        if glfw.get_key(self.window, glfw.KEY_LEFT) == glfw.PRESS:
            self.camera.rotate(-rotate_speed, 0)
        if glfw.get_key(self.window, glfw.KEY_RIGHT) == glfw.PRESS:
            self.camera.rotate(rotate_speed, 0)
        if glfw.get_key(self.window, glfw.KEY_UP) == glfw.PRESS:
            self.camera.rotate(0, rotate_speed)
        if glfw.get_key(self.window, glfw.KEY_DOWN) == glfw.PRESS:
            self.camera.rotate(0, -rotate_speed)

        if glfw.get_key(self.window, glfw.KEY_O) == glfw.PRESS:
            self.camera.zoom(rotate_speed * 10)
        if glfw.get_key(self.window, glfw.KEY_I) == glfw.PRESS:
            self.camera.zoom(-rotate_speed * 10)

        if glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS:
            xpos, ypos = glfw.get_cursor_pos(self.window)
            if self.first_mouse:
                self.first_mouse = False
                self.prev_x = xpos
                self.prev_y = ypos
            speed = 0.04 * rotate_speed
            xoffset = (xpos - self.prev_x) * speed
            yoffset = (ypos - self.prev_y) * speed
            self.camera.rotate(xoffset, yoffset)
        else:
            self.first_mouse = True

    def should_close(self):
        return glfw.window_should_close(self.window)

    def set_should_close(self, value):
        glfw.set_window_should_close(self.window, value)

    def begin_frame(self):

        t = time.time()
        self.frame_rate = t - self.current
        if (self.frame_rate != 0):
            fps = 1.0 / self.frame_rate
        self.duration += t - self.current
        self.current = t

        imgui.new_frame()
        self.frames += 1

        imgui.begin("Options")

        imgui.text("Time: {:.1f}".format(self.duration))
        imgui.text("FPS: {:.1f}".format(fps))
        imgui.text("Total Frames: {}".format(self.frames))

    # img: (height, width, 3) np.float32
    def write_texture_cpu(self, img):
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.texture_size[0], self.texture_size[1], GL_RGB, GL_FLOAT, img)
        glGenerateMipmap(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, 0)

    # img: (height, width, 3) torch.float32
    def write_texture_gpu(self, img):
        # flashing problem
        dr.sync_device()
        cres, = cudart.cudaGraphicsMapResources(1, self.bufobj, 0)
        check_cuda_error(cres)
        cres, ptr, size = cudart.cudaGraphicsResourceGetMappedPointer(self.bufobj)
        check_cuda_error(cres)
        cres, = cudart.cudaMemcpy(ptr, img.data_ptr(), size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)
        check_cuda_error(cres)
        cres, = cudart.cudaGraphicsUnmapResources(1, self.bufobj, 0)
        check_cuda_error(cres)

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, int(self.pbo))
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.texture_size[0],
                        self.texture_size[1], GL_RGB, GL_FLOAT, ctypes.c_void_p(0))
        glBindTexture(GL_TEXTURE_2D, 0)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0)

    def end_frame(self, save_path=None):
        save_image: bool = (save_path is not None) and (save_path != "")
        # calc_error: bool = (self.function_wrap != None) and \
        #     (self.function_wrap.get_should_calc_error())
        read_tex_in: bool = save_image #or calc_error

        if self.compute_task != None:
            self.compute_task.run(group_size=self.texture_size, tex_input=self.texture)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(self.program)
        glActiveTexture(GL_TEXTURE0)
        tex_input = self.texture
        if self.compute_task != None and hasattr(self.compute_task, "output_texture"):
            tex_input = self.compute_task.output_texture
        glBindTexture(GL_TEXTURE_2D, tex_input)
        glBindImageTexture(0, self.texture, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F)
        glUniform1i(glGetUniformLocation(self.program, "Image"), 0)  # binding in shader needs ogl420
        glUniform4f(glGetUniformLocation(self.program, "v1"), float(self.use_tonemapping), self.exposure, 0, 0)
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)

        if (read_tex_in):
            frame_buffer = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_FLOAT, None)
            frame_buffer = frame_buffer.reshape(self.height, self.width, 3)
            # frame_buffer = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_FLOAT)
            # frame_buffer = np.flip(frame_buffer.reshape(self.height, self.width, 3), 0)
            if (save_image):
                mi.util.write_bitmap(save_path, frame_buffer)
                dr.sync_all_devices()
            # if (calc_error):
            #     img = torch.from_numpy(frame_buffer.copy()).cuda()
            #     self.function_wrap.calc_error_run(img)

        imgui.end()

        imgui.render()
        imgui.end_frame()

        self.impl.render(imgui.get_draw_data())
        self.impl.process_inputs()
        self.process_input()
        glfw.swap_buffers(self.window)
        glfw.poll_events()

        glBindVertexArray(0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glUseProgram(0)

        # glh.check_errors()

    def check_and_update_texture_size(self, width, height):
        if width == self.texture_size[0] and height == self.texture_size[1]:
            return
        if self.gpu:
            if self.bufobj != None:
                cres, = cudart.cudaGraphicsUnregisterResource(self.bufobj)
                check_cuda_error(cres)
            if self.pbo != None:
                glDeleteBuffers(1, [self.pbo])

        if self.texture != None:
            glDeleteTextures(1, [self.texture])

        self.texture_size = (width, height)
        self.texture = self.create_texture(*self.texture_size)

        if self.gpu:
            self.pbo = self.create_pbo(self.texture_size[0], self.texture_size[1])
            cres, self.bufobj = cudart.cudaGraphicsGLRegisterBuffer(int(self.pbo), cudart.cudaGraphicsRegisterFlags(0))
            check_cuda_error(cres)

        # cue compute task
        if self.compute_task != None and (hasattr(self.compute_task, "resize")):
            self.compute_task.resize(*self.texture_size)

    # def set_compute_task(self, task: ComputeTask, release: bool = True):
    #     if self.compute_task != None and release:
    #         self.compute_task.release()
    #     self.compute_task = task