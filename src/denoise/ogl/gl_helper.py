from OpenGL.GL import *
from OpenGL.GL.ARB.pixel_buffer_object import *

import os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SHADER_DIR = os.path.join(CURRENT_DIR, '../shaders')


class OpenGLHelper(object):
    """OpenGL helper class."""

    @staticmethod
    def get_string(name):
        return glGetString(name).decode('utf-8')

    @staticmethod
    def get_integer(name):
        return glGetIntegerv(name)

    @staticmethod
    def get_integer_i(target, index):
        """indexed glGetIntegerv"""
        return glGetIntegeri_v(target, index)[0]

    @staticmethod
    def load_shader_file(path):
        path = os.path.join(SHADER_DIR, path)
        with open(path, 'r') as f:
            return f.read()

    @staticmethod
    def print_error(msg):
        print('\033[91m[Error]\033[00m {}'.format(msg))

    @staticmethod
    def check_compile_status(shader, compile: bool = False):
        ret = True
        if compile:
            if (glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE):
                err_str = glGetShaderInfoLog(shader)
                OpenGLHelper.print_error("[Compile] :{}".format(err_str.decode()))
                ret = False
        else:
            if glGetProgramiv(shader, GL_LINK_STATUS) != GL_TRUE:
                err_str = glGetProgramInfoLog(shader)
                OpenGLHelper.print_error("[Link] :{}".format(err_str.decode()))
                ret = False

    @staticmethod
    def create_compute_program(shader_path):
        shader_str = OpenGLHelper.load_shader_file(shader_path)
        return OpenGLHelper.create_compute_program_from_str(shader_str)

    @staticmethod
    def create_spv_compute_program(shader_path):
        # https://www.khronos.org/opengl/wiki/SPIR-V#SPIR-V_extensions
        # read binary spv file
        path = os.path.join(SHADER_DIR, shader_path)
        shader_bin = None
        with open(path, 'rb') as f:
            shader_bin = f.read()

        shader = glCreateShader(GL_COMPUTE_SHADER)
        glShaderBinary(1, [shader], GL_SHADER_BINARY_FORMAT_SPIR_V, shader_bin, len(shader_bin))
        glSpecializeShader(shader, "main", 0, None, None)
        if (glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE):
            err_str = glGetShaderInfoLog(shader)
            OpenGLHelper.print_error("[spv shader] :{}".format(err_str.decode()))

        program = glCreateProgram()
        glAttachShader(program, shader)
        glLinkProgram(program)
        OpenGLHelper.check_compile_status(program, False)
        # delete the shaders as they're linked into our program now and no longer necessary
        glDeleteShader(shader)

        return program

    @staticmethod
    def create_compute_program_from_str(shader_str: str):
        shader = glCreateShader(GL_COMPUTE_SHADER)
        glShaderSource(shader, shader_str)
        glCompileShader(shader)
        OpenGLHelper.check_compile_status(shader, True)

        program = glCreateProgram()
        glAttachShader(program, shader)
        glLinkProgram(program)
        OpenGLHelper.check_compile_status(program, False)
        # delete the shaders as they're linked into our program now and no longer necessary
        glDeleteShader(shader)
        return program

    @staticmethod
    def print_opengl_infos():
        print("\033[93mOpenGL Infos:\033[0m")
        print(" OpenGL Version: {}".format(OpenGLHelper.get_string(GL_VERSION)))
        print(" OpenGL Vendor: {}".format(OpenGLHelper.get_string(GL_VENDOR)))
        print(" OpenGL Renderer: {}".format(OpenGLHelper.get_string(GL_RENDERER)))
        print(" OpenGL Shading Language Version: {}".format(OpenGLHelper.get_string(GL_SHADING_LANGUAGE_VERSION)))
        print(" OpenGL Compute Shader Infos:")
        print("  Max Compute Work Group Count: (x, y, z) = ({}, {}, {})".format(
            OpenGLHelper.get_integer_i(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0),
            OpenGLHelper.get_integer_i(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 1),
            OpenGLHelper.get_integer_i(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 2))
        )
        print("  Max Compute Work Group Size: (x, y, z) = ({}, {}, {})".format(
            OpenGLHelper.get_integer_i(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0),
            OpenGLHelper.get_integer_i(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1),
            OpenGLHelper.get_integer_i(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2))
        )
        print("  Max Compute Work Group Invocations: {}".format(
            OpenGLHelper.get_integer(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS))
        )

    @staticmethod
    def check_errors():
        ret = False
        err = glGetError()
        if err != GL_NO_ERROR:
            print("OpenGL Error: {}".format(err))
            ret = True
        return ret
