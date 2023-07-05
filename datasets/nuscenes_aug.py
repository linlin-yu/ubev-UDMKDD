import os
import warnings

import torch.nn.functional as F
from diffusers import StableDiffusionInpaintPipeline
from shapely.errors import ShapelyDeprecationWarning
from xformers.ops import MemoryEfficientAttentionFlashAttentionOp

from datasets.nuscenes import *
from tools.geometry import *

warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)


class NuScenesDatasetAugmented(NuScenesDataset):
    def __init__(self, *args, device=4, **kwargs):
        super(NuScenesDatasetAugmented, self).__init__(*args, **kwargs)

        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float16,
        )

        self.pipe.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
        self.pipe.vae.enable_xformers_memory_efficient_attention(attention_op=None)
        self.pipe = self.pipe.to(device)

        self.prompts = [
            "Add a Brown bear, standing on the road, whole body visible, the bear should be large. photorealistic",
        ]

    def __getitem__(self, index):
        rec = self.ixes[index]
        images, intrinsics, extrinsics = self.get_input_data(rec)
        labels = self.get_label(rec)

        size = [2.25, 5., 2.]
        trans = [-8, 0., size[2] / 2]
        rot = euler_to_quaternion(0, 0, 0)

        cam = 1

        intrinsic = intrinsics[cam]
        extrinsic = np.linalg.inv(extrinsics[cam])

        bev_ood, cam_ood = render_ood(
            trans, rot, size,
            intrinsic, extrinsic,
            self.bev_resolution,
            self.bev_start_position,
            type='nuscenes',
        )

        sc = 2
        image_r = F.interpolate(images[None, cam], scale_factor=sc, mode='bilinear', align_corners=False)
        mask_r = F.interpolate(torch.tensor(cam_ood[None, None]),
                               scale_factor=sc, mode='bilinear', align_corners=False)

        result = torch.tensor(self.pipe(
            prompt=self.prompts[0],
            image=image_r * 2 - 1,
            mask_image=mask_r,
            width=480 * sc, height=224 * sc,
            output_type='np',
        ).images[0]).permute(2, 0, 1)[None]

        images[cam] = F.interpolate(result, scale_factor=1 / sc, mode='bilinear', align_corners=False)

        return images, intrinsics, extrinsics, labels, bev_ood, cam_ood


def compile_data(version, dataroot, batch_size=8, num_workers=16):
    nusc, dataroot = get_nusc(version, dataroot)

    train_data = NuScenesDatasetAugmented(nusc, True)
    val_data = NuScenesDatasetAugmented(nusc, False)

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
    )

    return train_loader, val_loader
