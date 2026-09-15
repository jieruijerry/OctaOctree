import os
import cv2
import numpy as np

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # Enable OpenEXR support in OpenCV

def linear_to_srgb(img):
    """Apply gamma correction (linear RGB -> sRGB)."""
    img = np.clip(img, 0, 1)
    return np.where(img <= 0.0031308,
                    12.92 * img,
                    1.055 * (img ** (1.0 / 2.4)) - 0.055)

def convert_exr_to_png(input_path, output_path, apply_srgb=True):
    img = cv2.imread(input_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    img = img ** (1 / 2.2)
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(output_path, img)

def batch_convert(folder_in, folder_out, apply_srgb=True):
    os.makedirs(folder_out, exist_ok=True)
    for file in os.listdir(folder_in):
        if file.lower().endswith(".exr"):
            input_path = os.path.join(folder_in, file)
            output_name = os.path.splitext(file)[0] + ".png"
            output_path = os.path.join(folder_out, output_name)
            print(f"Converting: {file} -> {output_name}")
            convert_exr_to_png(input_path, output_path, apply_srgb)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=str, required=True, help="Path to input EXR files")
    parser.add_argument("-o", "--output_dir", type=str, default="out/demo", help="Path to save PNGs")
    args = parser.parse_args()

    batch_convert(args.input_dir, args.output_dir, apply_srgb=False)