import math
import os
import random

import torch.nn.functional as F
from diffusers import StableDiffusionInpaintPipeline
from xformers.ops import MemoryEfficientAttentionFlashAttentionOp

from datasets.carla import CarlaDataset
from tools.geometry import *


def get_prompt(animal):
    return f"Add a {animal} standing on the road. The {animal} should be as large as possible. Make the {animal} photorealistic"


class CarlaDatasetAugmented(CarlaDataset):
    def __init__(self, *args, device=4, **kwargs):
        super(CarlaDatasetAugmented, self).__init__(*args, **kwargs)

        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float32,
        )

        self.pipe.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
        self.pipe.vae.enable_xformers_memory_efficient_attention(attention_op=None)
        self.pipe = self.pipe.to(device)

        self.animals = {
            "bear",
            "elephant",
            "horse",
            "deer"
        }

    def get_label(self, index, agent_path, bev_ood):
        label_r = Image.open(os.path.join(agent_path + "bev_semantic", f'{index}.png'))
        label = np.array(label_r)
        label_r.close()

        empty = np.ones(self.bev_dimension[:2])

        road = mask(label, (128, 64, 128))
        lane = mask(label, (157, 234, 50))
        vehicles = mask(label, (0, 0, 142))

        ood = mask(label, (0, 0, 0))
        bounding_boxes = find_bounding_boxes(ood)
        ood = draw_bounding_boxes(bounding_boxes)

        empty[vehicles == 1] = 0
        empty[road == 1] = 0
        empty[lane == 1] = 0
        label = np.stack((vehicles, road, lane, empty))

        return torch.tensor(label.copy()), torch.tensor(ood)

    def __getitem__(self, index):
        images, intrinsics, extrinsics, labels, ood = super().__getitem__(index)

        a = random.randrange(0, len(self.animals))

        size = [3., 5., 3.]  # w h d
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
            type='carla',
        )
        sc = 8
        image_r = F.interpolate(images[None, cam], scale_factor=sc, mode='bilinear', align_corners=False)
        mask_r = F.interpolate(torch.tensor(cam_ood[None, None]),
                               scale_factor=sc, mode='bilinear', align_corners=False)

        result = torch.tensor(self.pipe(
            prompt=get_prompt(self.animals[a]),
            image=image_r * 2 - 1,
            mask_image=mask_r,
            width=480 * sc, height=224 * sc,
            output_type='np',
        ).images[0]).permute(2, 0, 1)[None]

        images[cam] = F.interpolate(result, scale_factor=1 / sc, mode='bilinear', align_corners=False)

        return images, intrinsics, extrinsics, labels, bev_ood, cam_ood


def compile_data(version, dataroot, batch_size=8, num_workers=16, ood=False):
    train_data = CarlaDatasetAugmented(os.path.join(dataroot, "train"), True)
    val_data = CarlaDatasetAugmented(os.path.join(dataroot, "val"), False)

    if version == 'mini':
        train_sampler = torch.utils.data.RandomSampler(train_data, num_samples=128)
        val_sampler = torch.utils.data.RandomSampler(val_data, num_samples=128)

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True,
            sampler=train_sampler
        )

        val_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True,
            sampler=val_sampler
        )
    else:
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
