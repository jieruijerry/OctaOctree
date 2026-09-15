import os
import sys
import requests
import zipfile
import tarfile
import shutil
import argparse

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OIDN_DIR = os.path.join(CURRENT_DIR, "oidn")

# download libs
def download_libs():
    url = ""
    if sys.platform == "win32":
        url = "https://github.com/OpenImageDenoise/oidn/releases/download/v2.1.0/oidn-2.1.0.x64.windows.zip"
    elif sys.platform == "linux":
        url = "https://github.com/OpenImageDenoise/oidn/releases/download/v2.1.0/oidn-2.1.0.x86_64.linux.tar.gz"
    else:
        print("Unsupported platform!")
        exit(1)

    compress_file = url.split("/")[-1]
    target_file = os.path.join(OIDN_DIR, compress_file)

    # download file and move it to utils dir
    if not os.path.exists(target_file):
        print("Downloading oidn for {}...".format(sys.platform))
        r = requests.get(url)
        with open(compress_file, "wb") as code:
            code.write(r.content)

        if not os.path.exists(OIDN_DIR):
            os.mkdir(OIDN_DIR)
        os.rename(compress_file, target_file)
    else:
        print("Found oidn for {}...".format(sys.platform))

    # unzip file
    if sys.platform == "win32":
        # use python libs to unzip
        with zipfile.ZipFile(target_file, "r") as zip_ref:
            zip_ref.extractall(OIDN_DIR)  # unzip to current dir
    elif sys.platform == "linux":
        # use python libs to unzip
        with tarfile.open(target_file) as tar_ref:
            tar_ref.extractall(OIDN_DIR)  # unzip to current dir
    else:
        print("Unsupported platform!")
        exit(1)


def clean_oidn():
    if os.path.exists(OIDN_DIR):
        shutil.rmtree(OIDN_DIR)

def copy_pybind11():
    dst = os.path.join(CURRENT_DIR, "pybind11")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    src = os.path.join("{}/../../../".format(CURRENT_DIR), "external/pybind11")
    shutil.copytree(src, dst)

# main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true", help="clean oidn")
    args = parser.parse_args()

    if args.clean:
        clean_oidn()

    download_libs()

    copy_pybind11()
