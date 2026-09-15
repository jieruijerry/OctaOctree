import os
import subprocess
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(CURRENT_DIR, "../"))
sys.path.append(os.path.join(CURRENT_DIR, "../../"))

from denoise.config import check_user_settings
check_user_settings()
import shutil

# gen *.pyd at . location
cmd = [
    sys.executable,
    "-m",
    "pip",
    "install",
    "--no-build-isolation",
    CURRENT_DIR,
]
subprocess.check_call(cmd)

# from simple_denoise.prepare_shaders import generate_OpenGL_shaders
# generate_OpenGL_shaders()
