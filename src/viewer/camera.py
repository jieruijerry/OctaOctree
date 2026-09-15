import numpy as np

from typing import List

class FPSCamera:

    # np.ndarray (4, 4)
    def __init__(self, intrinsics, extrinsics, speed):
        # Intrinsics
        self.width = intrinsics['width']
        self.height = intrinsics['height']

        self.set_x_fov(intrinsics['x_fov'])
        self.set_transform(extrinsics)
        self.speed = speed

        # Quaternion of rotation
        self.quat = None

    def get_x_fov(self):
        return self.x_fov

    def get_transform(self):
        return np.array([
            [self.x[0], self.y[0], self.z[0], self.pos[0]],
            [self.x[1], self.y[1], self.z[1], self.pos[1]],
            [self.x[2], self.y[2], self.z[2], self.pos[2]],
            [0, 0, 0, 1]
        ])
    
    def get_quaternion(self):
        """
        Get quaternion representation of rotation
        """
        if self.quat is not None:
            return self.quat
        
        matrix = self.get_transform()
        trace = matrix[0, 0] + matrix[1, 1] + matrix[2, 2]

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            qw = 0.25 / s
            qx = (matrix[2, 1] - matrix[1, 2]) * s
            qy = (matrix[0, 2] - matrix[2, 0]) * s
            qz = (matrix[1, 0] - matrix[0, 1]) * s
        else:
            if matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
                s = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
                qw = (matrix[2, 1] - matrix[1, 2]) / s
                qx = 0.25 * s
                qy = (matrix[0, 1] + matrix[1, 0]) / s
                qz = (matrix[0, 2] + matrix[2, 0]) / s
            elif matrix[1, 1] > matrix[2, 2]:
                s = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
                qw = (matrix[0, 2] - matrix[2, 0]) / s
                qx = (matrix[0, 1] + matrix[1, 0]) / s
                qy = 0.25 * s
                qz = (matrix[1, 2] + matrix[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
                qw = (matrix[1, 0] - matrix[0, 1]) / s
                qx = (matrix[0, 2] + matrix[2, 0]) / s
                qy = (matrix[1, 2] + matrix[2, 1]) / s
                qz = 0.25 * s

        quaternion = np.array([qx, qy, qz, qw])

        # Normalize the quaternion
        quaternion /= np.linalg.norm(quaternion)

        # Cache the quaternion
        self.quat = quaternion

        return quaternion
    
    def set_x_fov(self, x_fov):
        self.x_fov = x_fov

    def set_transform(self, extrinsics):
        self.pos = extrinsics[:3, 3]
        self.x = extrinsics[:3, 0]
        self.y = extrinsics[:3, 1]
        self.z = extrinsics[:3, 2]

        self.phi = np.rad2deg(np.arctan2(self.z[2], self.z[0]))
        self.theta = np.rad2deg(np.arccos(self.z[1]))

    # idx: 0 for z, 1 for y, 2 for x
    def move(self, idx, delta):
        if idx == 0:
            self.pos += delta * self.z * self.speed
        elif idx == 1:
            self.pos += delta * self.y * self.speed
        else:
            self.pos += delta * self.x * self.speed

    def rotate(self, x, y):
        self.theta = np.clip(self.theta + y, 1, 179)
        self.phi += x

        cos_theta = np.cos(np.deg2rad(self.theta))
        sin_theta = np.sin(np.deg2rad(self.theta))
        cos_phi = np.cos(np.deg2rad(self.phi))
        sin_phi = np.sin(np.deg2rad(self.phi))

        self.z = np.array(
            [sin_theta * cos_phi, cos_theta, sin_theta * sin_phi])
        self.x = np.array([sin_phi, 0, -cos_phi])
        self.y = np.array(
            [-cos_theta * cos_phi, sin_theta, -cos_theta * sin_phi])
    
    def zoom(self, delta):
        self.x_fov = np.clip(self.x_fov + delta * self.speed, 10, 90)


def quaternion_slerp(q1, q2, t):
    """
    Quaternion Spherical Linear Interpolation
    """
    cos_theta = np.dot(q1, q2)

    if cos_theta < 0:
        q1 = -q1
        cos_theta = -cos_theta
    
    angle = np.arccos(cos_theta)
    return (np.sin((1 - t) * angle) * q1 + np.sin(t * angle) * q2) / np.sin(angle)

def quat2Matrix(rot, pos):
    """
    Convert quaternion to transformation matrix (Left-handed)
    """
    xx = rot[0] * rot[0]
    yy = rot[1] * rot[1]
    zz = rot[2] * rot[2]
    xy = rot[0] * rot[1]
    xz = rot[0] * rot[2]
    yz = rot[1] * rot[2]
    xw = rot[0] * rot[3]
    yw = rot[1] * rot[3]
    zw = rot[2] * rot[3]
    
    return np.array([
        [1 - 2 * (yy + zz), 2 * (xy - zw), 2 * (xz + yw), pos[0]],
        [2 * (xy + zw), 1 - 2 * (xx + zz), 2 * (yz - xw), pos[1]],
        [2 * (xz - yw), 2 * (yz + xw), 1 - 2 * (xx + yy), pos[2]],
        [0, 0, 0, 1]
    ])

class MovingCamera:
    def __init__(self, cameras: List[FPSCamera]):
        self.cameras = cameras
    
    def get_camera(self, section, cam_v) -> FPSCamera:
        # Get two cameras
        cam1: FPSCamera = self.cameras[section]
        cam2: FPSCamera = self.cameras[(section + 1) % len(self.cameras)]

        assert cam1.width == cam2.width, "Width mismatch"
        assert cam1.height == cam2.height, "Height mismatch"

        # Interpolate extrinsics
        pos = cam1.pos + (cam2.pos - cam1.pos) * cam_v
        rot1 = cam1.get_quaternion()
        rot2 = cam2.get_quaternion()
        # print("t", cam_v)
        # print("rot1", rot1)
        # print("rot2", rot2)
        rot = quaternion_slerp(rot1, rot2, cam_v)
        # print("rot", rot)
        extrinsics = quat2Matrix(rot, pos)

        # Interpolate intrinsics
        x_fov = cam1.x_fov + (cam2.x_fov - cam1.x_fov) * cam_v

        return FPSCamera({
            "width": cam1.width,
            "height": cam1.height,
            "x_fov": x_fov
        }, extrinsics, cam1.speed)

