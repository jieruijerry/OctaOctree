import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
VENDORED_PYBIND11 = os.path.abspath(os.path.join(CURRENT_DIR, "../../../external/pybind11"))
if os.path.isdir(VENDORED_PYBIND11):
    sys.path.insert(0, VENDORED_PYBIND11)

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "pyNISConfigWrapper",
        ["cpp/bind.cpp"],
    ),
]

setup(
    name="pyNISConfigWrapper",
    ext_modules=ext_modules,
    version="0.0.1",
    extras_require={"test": "pytest"},
    # Currently, build_ext only provides an optional "highest supported C++
    # level" feature, but in the future it may provide more features.
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
)
