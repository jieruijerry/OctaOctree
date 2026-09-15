# glob all the install.py files in the subdirectories and run them
import os
import glob
import subprocess

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

import sys
sys.path.append(os.path.join(CURRENT_DIR, "../"))
from denoise.config import check_user_settings

def install_all():
    install_files = glob.glob(os.path.join(CURRENT_DIR, "**", "install.py"), recursive=True)

    print("    {}".format("\n    ".join(install_files)))
    for install_file in install_files:
        cmd = [sys.executable, install_file]
        # print cmd in green style
        print("Running: \033[92m{}\033[00m".format(" ".join(cmd)))
        subprocess.check_call(cmd)

if __name__ == "__main__":
    check_user_settings()
    install_all()
