# find all the build directoris in the src directory and remove them

import os
import shutil
import glob

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def clean_build():
    to_delete = ["./**/build", "./**/__pycache__", "./**/dist", "./**/*.egg-info"]
    for i in to_delete:
        files = glob.glob(i, recursive=True)
        for f in files:
            f = os.path.join(CURRENT_DIR, f)
            if not os.path.exists(f):
                continue
            if os.path.isdir(f):
                shutil.rmtree(f)
            else:
                os.remove(f)


def uninstall_libs():
    libs = ["setup_optix_example", "setup_oidn_example", "cuda_extension"]
    cmd = "pip uninstall"
    for lib in libs:
        os.system("{} {}".format(cmd, lib))


if __name__ == "__main__":
    clean_build()
    uninstall_libs()
