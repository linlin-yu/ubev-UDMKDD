from PIL import Image
import torch
import numpy as np


def resize_and_crop_image(img, resize_dims, crop):
    img = img.resize(resize_dims, resample=Image.BILINEAR)
    img = img.crop(crop)
    return img


def mask(img, target):
    m = np.all(img == target, axis=2).astype(int)
    return m


def inverse_extrinsics(E):
    R = E[0:3, 0:3]
    T = E[0:3, 3].reshape(3, 1)

    R_inv = np.transpose(R)

    T_inv = -np.dot(R_inv, T)

    E_inv = np.identity(4)
    E_inv[0:3, 0:3] = R_inv
    E_inv[0:3, 3] = T_inv[:, 0]

    return E_inv


def update_intrinsics(intrinsics, top_crop=0.0, left_crop=0.0, scale_width=1.0, scale_height=1.0):
    updated_intrinsics = intrinsics.clone()

    updated_intrinsics[0, 0] *= scale_width
    updated_intrinsics[0, 2] *= scale_width
    updated_intrinsics[1, 1] *= scale_height
    updated_intrinsics[1, 2] *= scale_height

    updated_intrinsics[0, 2] -= left_crop
    updated_intrinsics[1, 2] -= top_crop

    return updated_intrinsics


def calculate_birds_eye_view_parameters(x_bounds, y_bounds, z_bounds):
    bev_resolution = torch.tensor([row[2] for row in [x_bounds, y_bounds, z_bounds]])
    bev_start_position = torch.tensor([row[0] + row[2] / 2.0 for row in [x_bounds, y_bounds, z_bounds]])
    bev_dimension = torch.tensor([(row[1] - row[0]) / row[2] for row in [x_bounds, y_bounds, z_bounds]],
                                 dtype=torch.long)

    return bev_resolution, bev_start_position, bev_dimension


def warp_features(x, flow, mode='nearest', spatial_extent=None):
    if flow is None:
        return x
    b, c, h, w = x.shape
    angle = flow[:, 5].clone()
    translation = flow[:, :2].clone()

    translation[:, 0] /= spatial_extent[0]
    translation[:, 1] /= spatial_extent[1]

    translation[:, 0] *= -1

    cos_theta = torch.cos(angle)
    sin_theta = torch.sin(angle)

    transformation = torch.stack([cos_theta, -sin_theta, translation[:, 1],
                                  sin_theta, cos_theta, translation[:, 0]], dim=-1).view(b, 2, 3)

    grid = torch.nn.functional.affine_grid(transformation, size=x.shape, align_corners=False)
    warped_x = torch.nn.functional.grid_sample(x, grid.float(), mode=mode, padding_mode='zeros', align_corners=False)

    return warped_x


def bounding_box(points):
    x_coordinates, y_coordinates = zip(*points)

    return min(x_coordinates), min(y_coordinates), max(x_coordinates), max(y_coordinates)
