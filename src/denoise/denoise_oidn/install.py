import os
import subprocess
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# download oidn
subprocess.check_call([sys.executable, os.path.join(CURRENT_DIR, "download_resources.py")])

# install oidn
subprocess.check_call([
    sys.executable,
    "-m",
    "pip",
    "install",
    "--no-build-isolation",
    CURRENT_DIR,
])
