from yacs.config import CfgNode as CN
import os
import sys

_C = CN()

############################# Personal Settings ########################
OPTIX_SDK_PATH = ""
CUDA_PATH = ""
CL_PATH = ""

if sys.platform == "win32":
    CL_PATH = "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.38.33130/bin/Hostx64/x64"
elif sys.platform == "linux":
    user_home = os.path.expanduser('~')
    # gcc
    CL_PATH = "/usr/bin"
    OPTIX_SDK_PATH = "{}/soft/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/".format(
        user_home)
    CUDA_PATH = "/usr/local/cuda-12.1"
############################# Personal Settings ########################


if (OPTIX_SDK_PATH == ""):
    OPTIX_SDK_PATH = os.environ["OptiX_INSTALL_DIR"]
if (CUDA_PATH == ""):
    CUDA_PATH = os.environ["CUDA_PATH"]

INCLUDE_PATHS = []
INCLUDE_PATHS.append(os.path.join(OPTIX_SDK_PATH, "include"))
INCLUDE_PATHS.append(os.path.join(CUDA_PATH, "include"))

_C.INCLUDE_PATHS = ";".join(INCLUDE_PATHS)
_C.INCLUDE_PATHS = _C.INCLUDE_PATHS.replace("\\", "/")
_C.CUDA_PATH = CUDA_PATH
_C.PATH = CL_PATH

_C.VK_PATH = ""


def check_user_settings():
    print(
        "\033[93mPlease check your optix settings in the {}\033[00m".format(__file__))
    print("    OptiX SDK Path: {}".format(OPTIX_SDK_PATH))
    print("    Cuda Path: {}".format(CUDA_PATH))
    print("    GCC Path: {}".format(CL_PATH))
    print()
