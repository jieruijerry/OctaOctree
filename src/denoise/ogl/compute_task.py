from OpenGL.GL import *
import os
import sys
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(CURRENT_DIR)
from gl_helper import OpenGLHelper as glh

class ComputeTask(object):
    def __init__(self, name, shader_path):
        self.name = name
        self.program = glh.create_compute_program(shader_path)

    def run(self, **kwargs):
        raise NotImplementedError

    def render_ui(self):
        raise NotImplementedError

    def set(self, **kwargs):
        raise NotImplementedError

    def release(self):
        glDeleteProgram(self.program)