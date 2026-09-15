import os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

import sys
sys.path.append(os.path.join(CURRENT_DIR, "../"))
sys.path.append(os.path.join(CURRENT_DIR, "../../"))

from denoise.config import _C as cfg
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

__version__ = "0.0.1"

os.environ['PATH'] = os.environ['PATH'] + os.pathsep + cfg.PATH

# compile flags
c_flags = []
if sys.platform == "win32":
    c_flags.extend(["/DEBUG:NONE", "/O2"])
elif sys.platform == "linux":
    c_flags.extend(["-O3"])

ld_flags = []
if sys.platform == "win32":
    ld_flags.append("/NODEFAULTLIB:LIBCMT")
    ld_flags.append("/LIBPATH:{}/lib/x64/".format(cfg.CUDA_PATH))
    ld_flags.append('cuda.lib')
    ld_flags.append('cudart_static.lib')
    # ld_flags.append('advapi32.lib')
elif sys.platform == "linux":
    ld_flags.append("-L{}/lib64/stubs/".format(cfg.CUDA_PATH))
    ld_flags.append("-lcuda")

platform_dir = ""
if sys.platform == "win32":
    platform_dir = "oidn-2.1.0.x64.windows"
elif sys.platform == "linux":
    platform_dir = "oidn-2.1.0.x86_64.linux"
else:
    raise NotImplementedError

oidn_include_dir = os.path.join(CURRENT_DIR, "oidn/{}/include".format(platform_dir))
oidn_include_dir = [oidn_include_dir]
oidn_include_dir.extend(cfg.INCLUDE_PATHS.split(";"))

oidn_lib_dir = os.path.join(CURRENT_DIR, "oidn/{}/lib".format(platform_dir))
if sys.platform == "win32":
    ld_flags.append("/LIBPATH:{}".format(oidn_lib_dir))
    ld_flags.append("OpenImageDenoise.lib")
    ld_flags.append("OpenImageDenoise_core.lib")
elif sys.platform == "linux":
    ld_flags.append("{}/{}".format(oidn_lib_dir, "libOpenImageDenoise.so.2.1.0"))
    # dynamic library load path
    ld_flags.append("-Wl,-rpath={}".format(oidn_lib_dir))

ext_modules = [
    CUDAExtension(
        name='setup_oidn_example',
        sources=['bind.cpp', 'oidn_denoiser.cpp', "utils.cpp"],
        include_dirs=oidn_include_dir,
        extra_compile_args={'cxx': c_flags},
        extra_link_args=ld_flags,
    ),
]

setup(
    name="setup_oidn_example",
    version=__version__,
    author="banbao990",
    description="oidn denoiser example",
    long_description="",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
    extras_require={"test": ["pytest>=6.0"]},
    python_requires=">=3.9",
)
