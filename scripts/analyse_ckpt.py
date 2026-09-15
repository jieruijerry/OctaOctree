from collections import defaultdict
import torch
import argparse


def human(n):
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TB"


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_ckpt", type=str, required=True, help="Path to input checkpoint file")
    args = parser.parse_args()

    group_bytes = defaultdict(int)
    ckpt = torch.load(args.input_ckpt, map_location="cpu")
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    # print(sd.keys())

    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        nbytes = v.numel() * v.element_size()

        # if k.startswith("encoding.octree.keys."):
        #     group = "encoding.octree.keys"
        # elif k.startswith("encoding.octree.children."):
        #     group = "encoding.octree.children"
        # elif k.startswith("encoding.octree.neighs."):
        #     group = "encoding.octree.neighs"
        # else:
        group = k.rsplit(".", 1)[0]

        group_bytes[group] += nbytes

    for k, nbytes in sorted(group_bytes.items(), key=lambda x: x[1], reverse=True):
        print(f"{human(nbytes):>10}  {k}")
