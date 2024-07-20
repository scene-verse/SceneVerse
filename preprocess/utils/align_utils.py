import numpy as np
import math

def compute_box_3d(size, center, rotmat):
    """Compute corners of a single box from rotation matrix
    Args:
        size: list of float [dx, dy, dz]
        center: np.array [x, y, z]
        rotmat: np.array (3, 3)
    Returns:
        corners: (8, 3)
    """
    l, h, w = [i / 2 for i in size]
    center = np.reshape(center, (-1, 3))
    center = center.reshape(3)
    x_corners = [l, l, -l, -l, l, l, -l, -l]
    y_corners = [h, -h, -h, h, h, -h, -h, h]
    z_corners = [w, w, w, w, -w, -w, -w, -w]
    corners_3d = np.dot(
        np.transpose(rotmat), np.vstack([x_corners, y_corners, z_corners])
    )
    corners_3d[0, :] += center[0]
    corners_3d[1, :] += center[1]
    corners_3d[2, :] += center[2]
    return np.transpose(corners_3d)


def rotate_z_axis_by_degrees(pointcloud, theta, clockwise=True):
    theta = np.deg2rad(theta)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    rot_matrix = np.array([[cos_t, -sin_t, 0],
                           [sin_t, cos_t, 0],
                           [0, 0, 1]], pointcloud.dtype)
    if not clockwise:
        rot_matrix = rot_matrix.T
    return pointcloud.dot(rot_matrix)


def eulerAnglesToRotationMatrix(theta):
    """Euler rotation matrix with clockwise logic.
    Rotation

    Args:
        theta: list of float
            [theta_x, theta_y, theta_z]
    Returns:
        R: np.array (3, 3)
            rotation matrix of Rz*Ry*Rx
    """
    R_x = np.array(
        [
            [1, 0, 0],
            [0, math.cos(theta[0]), -math.sin(theta[0])],
            [0, math.sin(theta[0]), math.cos(theta[0])],
        ]
    )

    R_y = np.array(
        [
            [math.cos(theta[1]), 0, math.sin(theta[1])],
            [0, 1, 0],
            [-math.sin(theta[1]), 0, math.cos(theta[1])],
        ]
    )

    R_z = np.array(
        [
            [math.cos(theta[2]), -math.sin(theta[2]), 0],
            [math.sin(theta[2]), math.cos(theta[2]), 0],
            [0, 0, 1],
        ]
    )

    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


def is_axis_aligned(rotated_box, thres=0.05):
    x_diff = abs(rotated_box[0][0] - rotated_box[1][0])
    y_diff = abs(rotated_box[0][1] - rotated_box[3][1])
    return x_diff < thres and y_diff < thres


def calc_align_matrix(bbox_list):
    RANGE = [-45, 45]
    NUM_BIN = 90
    angles = np.linspace(RANGE[0], RANGE[1], NUM_BIN)
    angle_counts = {}
    for _a in angles:
        bucket = round(_a, 3)
        for box in bbox_list:
            box_r = rotate_z_axis_by_degrees(box, bucket)
            bottom = box_r[4:]
            if is_axis_aligned(bottom):
                angle_counts[bucket] = angle_counts.get(bucket, 0) + 1
    if len(angle_counts) == 0:
        RANGE = [-90, 90]
        NUM_BIN = 180
        angles = np.linspace(RANGE[0], RANGE[1], NUM_BIN)
        for _a in angles:
            bucket = round(_a, 3)
            for box in bbox_list:
                box_r = rotate_z_axis_by_degrees(box, bucket)
                bottom = box_r[4:]
                if is_axis_aligned(bottom, thres=0.15):
                    angle_counts[bucket] = angle_counts.get(bucket, 0) + 1
    most_common_angle = max(angle_counts, key=angle_counts.get)
    return most_common_angle
