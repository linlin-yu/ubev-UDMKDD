from nuscenes.utils.geometry_utils import (box_in_image,
                                           view_points, BoxVisibility)

from datasets.nuscenes import *
from tools.geometry import *

import random
import matplotlib.path as mpltPath

from diffusers import StableDiffusionInpaintPipeline
from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
from scipy.spatial import ConvexHull

import torch.nn.functional as F


def fill_convex_hull(image, points, fill_value=1.0):

    hull = ConvexHull(points)
    hull_points = points[hull.vertices]

    pts = np.array(hull_points, 'int32')
    pts = pts.reshape((-1, 1, 2))

    cv2.fillPoly(image, [pts], fill_value)


class NuScenesDatasetOOD(NuScenesDataset):
    def __init__(self, *args, device=4, **kwargs):
        super(NuScenesDatasetOOD, self).__init__(*args, **kwargs)

        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float16,
        )
        
        self.pipe.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
        self.pipe.vae.enable_xformers_memory_efficient_attention(attention_op=None)
        self.pipe = self.pipe.to(device)

        self.prompts = [
            "Add a Brown bear, standing on the road, whole body visible, the bear should be as large as possible. photorealistic",
        ]

    def __getitem__(self, index):
        rec = self.ixes[index]

        images, intrinsics, extrinsics = self.get_input_data(rec)
        labels = self.get_label(rec)

        self.nusc.get('ego_pose',
                      self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])

        ood = np.zeros(self.bev_dimension[:2])

        trans = [0., 0., 0.]
        size = [5., 5., 3]
        rot = [0., 0., 0., 1.]

        if random.random() < 0.5:
            cam = 1
            trans[0] = random.uniform(13, 18)
        else:
            cam = 4
            trans[0] = random.uniform(-18, -13)
        trans[1] = random.uniform(-2, 2)

        box = Box(trans, size, Quaternion(rot))
        pts = box.bottom_corners()[:2].T
        pts = np.round(
            (pts - self.bev_start_position[:2] + self.bev_resolution[:2] / 2.0) / self.bev_resolution[:2]
        ).astype(np.int32)
        pts[:, [1, 0]] = pts[:, [0, 1]]
        cv2.fillPoly(ood, [pts], 1.0)

        cam_oods = np.zeros((len(self.cameras), 224, 480))

        crec = self.nusc.get('sample_data', rec['data'][self.cameras[cam]])
        cs_record = self.nusc.get('calibrated_sensor', crec['calibrated_sensor_token'])

        cb = box.copy()
        cb.translate(-np.array(cs_record['translation']))
        cb.rotate(Quaternion(cs_record['rotation']).inverse)

        corners = view_points(cb.corners(), intrinsics[cam], normalize=True)[:2, :]
        corners = np.int32(corners).T

        fill_convex_hull(cam_oods[cam], corners)

        sc = 2

        image_r = F.interpolate(images[None, cam], scale_factor=sc, mode='bilinear', align_corners=False)
        mask_r = F.interpolate(torch.tensor(cam_oods[None, None, cam]), scale_factor=sc, mode='bilinear', align_corners=False)

        result = torch.tensor(self.pipe(
            prompt=self.prompts[0],
            image=image_r * 2 - 1,
            mask_image=mask_r,
            width=480*sc, height=224*sc,
            output_type='np',
            strength=1.,
            num_inference_steps=50,
        ).images[0]).permute(2, 0, 1)[None]

        images[cam] = F.interpolate(result, scale_factor=1 / sc, mode='bilinear', align_corners=False)

        return images, intrinsics, extrinsics, labels, ood, cam_oods


def compile_data(version, dataroot, batch_size=8, num_workers=16):
    nusc, dataroot = get_nusc(version, dataroot)

    train_data = NuScenesDatasetOOD(nusc, True)
    val_data = NuScenesDatasetOOD(nusc, False)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    return train_loader, val_loader
