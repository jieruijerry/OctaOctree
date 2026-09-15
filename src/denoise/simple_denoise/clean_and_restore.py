import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append("{}/../".format(CURRENT_DIR))

from utils.clean import clean_build_and_uninstall

if __name__ == "__main__":
    lib_name = "pyNISConfigWrapper"
    clean_build_and_uninstall(lib_name, CURRENT_DIR)
