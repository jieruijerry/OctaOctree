import argparse
from typing import Optional, Sequence, Union

import numpy as np


NumberInput = Union[str, Sequence[float], np.ndarray]


IDENTITY_4X4 = "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"


def _parse_floats(value: NumberInput, expected: int, name: str) -> np.ndarray:
    if isinstance(value, str):
        parts = value.split()
    else:
        parts = list(value)

    if len(parts) != expected:
        raise ValueError(f"{name} expects {expected} values, got {len(parts)}")

    try:
        return np.array([float(part) for part in parts], dtype=np.float64)
    except ValueError as exc:
        raise ValueError(f"{name} contains a non-numeric value") from exc


def _as_matrix4x4(matrix: Optional[NumberInput]) -> np.ndarray:
    if matrix is None or (isinstance(matrix, str) and not matrix.strip()):
        return np.eye(4, dtype=np.float64)
    return _parse_floats(matrix, 16, "matrix").reshape(4, 4)


def _scale_matrix(scale: NumberInput) -> np.ndarray:
    sx, sy, sz = _parse_floats(scale, 3, "scale")
    return np.array(
        [
            [sx, 0.0, 0.0, 0.0],
            [0.0, sy, 0.0, 0.0],
            [0.0, 0.0, sz, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _rotation_matrix(axis: NumberInput, angle_degrees: float) -> np.ndarray:
    axis_vec = _parse_floats(axis, 3, "axis")
    norm = np.linalg.norm(axis_vec)

    if norm == 0.0:
        if angle_degrees == 0.0:
            return np.eye(4, dtype=np.float64)
        raise ValueError("axis must be non-zero when angle is not 0")

    x, y, z = axis_vec / norm
    theta = np.deg2rad(angle_degrees)
    c = np.cos(theta)
    s = np.sin(theta)
    one_minus_c = 1.0 - c

    return np.array(
        [
            [c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s, 0.0],
            [y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s, 0.0],
            [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _translation_matrix(translation: NumberInput) -> np.ndarray:
    tx, ty, tz = _parse_floats(translation, 3, "translation")
    return np.array(
        [
            [1.0, 0.0, 0.0, tx],
            [0.0, 1.0, 0.0, ty],
            [0.0, 0.0, 1.0, tz],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def local2world(
    matrix: Optional[NumberInput] = None,
    scale: NumberInput = "1 1 1",
    axis: NumberInput = "0 1 0",
    angle: float = 0.0,
    translation: NumberInput = "0 0 0",
) -> np.ndarray:
    """Apply scale, then rotation, then translation."""
    input_matrix = _as_matrix4x4(matrix)
    transform = _translation_matrix(translation) @ _rotation_matrix(axis, float(angle)) @ _scale_matrix(scale)
    return input_matrix @ transform


def flatten_matrix(matrix: np.ndarray, order: str = "row") -> str:
    if order not in ("row", "col", "column"):
        raise ValueError('order must be "row", "col", or "column"')

    matrix = np.where(np.abs(matrix) < 1e-12, 0.0, matrix)
    flatten_order = "F" if order in ("col", "column") else "C"
    return " ".join(format(value, ".12g") for value in matrix.reshape(-1, order=flatten_order))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute a flattened local-to-world transform matrix."
    )
    parser.add_argument(
        "-m",
        "--matrix",
        default=IDENTITY_4X4,
        help="Flattened 4x4 input matrix, separated by spaces.",
    )
    parser.add_argument(
        "-s",
        "--scale",
        default="1 1 1",
        help='Scale vector, separated by spaces. Default: "1 1 1".',
    )
    parser.add_argument(
        "-a",
        "--axis",
        default="0 1 0",
        help='Rotation axis, separated by spaces. Default: "0 1 0".',
    )
    parser.add_argument(
        "-r",
        "--angle",
        type=float,
        default=0.0,
        help="Rotation angle in degrees. Default: 0.",
    )
    parser.add_argument(
        "-t",
        "--translation",
        default="0 0 0",
        help='Translation vector, separated by spaces. Default: "0 0 0".',
    )
    parser.add_argument(
        "-o",
        "--output-order",
        choices=("row", "col", "column"),
        default="row",
        help='Flatten output order. Use "row" for row-major or "col"/"column" for column-major. Default: "row".',
    )
    parser.add_argument(
        "-i",
        "--inverse",
        action="store_true",
        help="Output the inverse transform matrix.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = local2world(
        matrix=args.matrix,
        scale=args.scale,
        axis=args.axis,
        angle=args.angle,
        translation=args.translation,
    )
    if args.inverse:
        result = np.linalg.inv(result)
    print(flatten_matrix(result, args.output_order))


if __name__ == "__main__":
    main()
