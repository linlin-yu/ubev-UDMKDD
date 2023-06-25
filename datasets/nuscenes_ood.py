from nuscenes.utils.geometry_utils import (box_in_image,
                                           view_points, BoxVisibility)

from datasets.nuscenes import *
from tools.geometry import *

import random

from diffusers import StableDiffusionInpaintPipeline
from xformers.ops import MemoryEfficientAttentionFlashAttentionOp

import torch.nn.functional as F


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

        trans = [0., random.uniform(-2, 2), 0.]
        size = [5., 5., 3]
        rot = [0., 0., 0., 1.]

        if random.random() < 0.5:
            cam = 1
            trans[0] = random.uniform(13, 18)
        else:
            cam = 4
            trans[0] = random.uniform(-18, -13)

        bev_ood, cam_ood = render_ood(
            trans, rot, size,
            intrinsics[cam], inverse_extrinsics(extrinsics[cam]),
            self.bev_resolution,
            self.bev_start_position
        )

        print(extrinsics[cam])

        cam_oods = np.zeros((6, 224, 480))
        cam_oods[cam] = cam_ood

        sc = 2

        image_r = F.interpolate(images[None, cam], scale_factor=sc, mode='bilinear', align_corners=False)
        mask_r = F.interpolate(torch.tensor(cam_ood[None, None]),
                               scale_factor=sc, mode='bilinear', align_corners=False)

        print(np.sum(cam_ood))
        result = torch.tensor(self.pipe(
            prompt=self.prompts[0],
            image=image_r * 2 - 1,
            mask_image=mask_r,
            width=480 * sc, height=224 * sc,
            output_type='np',
            strength=1.,
            num_inference_steps=50,
        ).images[0]).permute(2, 0, 1)[None]

        images[cam] = F.interpolate(result, scale_factor=1 / sc, mode='bilinear', align_corners=False)

        return images, intrinsics, extrinsics, labels, bev_ood, cam_oods


def compile_data(version, dataroot, batch_size=8, num_workers=16):
    nusc, dataroot = get_nusc(version, dataroot)

    train_data = NuScenesDatasetOOD(nusc, True)
    val_data = NuScenesDatasetOOD(nusc, False)

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )

    return train_loader, val_loader
