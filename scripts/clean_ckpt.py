import os
import argparse

import torch


def clean_ckpt(args):
    ckpt = torch.load(args.input_ckpt)

    keys_to_remove = [
        "optimizer_states",
        "lr_schedulers",
        "callbacks",
    ]

    for key in keys_to_remove:
        ckpt.pop(key, None)

    torch.save(ckpt, args.input_ckpt + ".cleaned")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_ckpt", type=str, required=True, help="Path to input checkpoint file")
    args = parser.parse_args()

    clean_ckpt(args)