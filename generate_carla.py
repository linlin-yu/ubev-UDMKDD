from datasets.carla import *
from tools.utils import *
import random
from time import time
from tqdm.notebook import tqdm

import torch.nn.functional as F
import matplotlib.pyplot as plt

from diffusers import StableDiffusionInpaintPipeline
from xformers.ops import MemoryEfficientAttentionFlashAttentionOp


def get_prompt(animal):
    return f"Add a {animal} standing on the road. The {animal} should be as large as possible. Make the {animal} photorealistic"


def generate():
    data_path = "../data/carla/train"
    carla_data = CarlaDataset(data_path, False)

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16,
    )

    pipe.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
    pipe.vae.enable_xformers_memory_efficient_attention(attention_op=None)
    pipe = pipe.to(0)

    animals = [
        "bear",
        "elephant",
        "horse",
        "deer"
    ]

    cameras = [
        'left_front_camera',
        'front_camera',
        'right_front_camera',
        'left_back_camera',
        'back_camera',
        'right_back_camera'
    ]

    save_dir = "../../data/carla/train_aug"

    for i, (images, intrinsics, extrinsics, labels, ood) in enumerate(tqdm(carla_data)):
        agent_number = math.floor(i / carla_data.ticks)
        agent_path = os.path.join(data_path, f"agents/{agent_number}/")
        save_path = os.path.join(save_dir, f"agents/{agent_number}/")
        index = (i + carla_data.offset) % carla_data.ticks
        os.makedirs(os.path.join(save_path, 'bev_semantic'), exist_ok=True)

        size = [3., 5., 3.]
        trans = [random.randint(8, 12), random.randint(-3, 3), size[2] / 2]
        rot = euler_to_quaternion(0, 0, 0)
        cam = 4
        a = random.randrange(0, len(animals))

        if random.choice([True, False]):
            trans[0] *= -1
            cam = 1

        label_r = Image.open(os.path.join(agent_path + "bev_semantic", f'{index}.png'))
        label = np.array(label_r)
        label_r.close()

        intrinsic = intrinsics[cam]
        extrinsic = np.linalg.inv(extrinsics[cam])

        bev_ood, cam_ood = render_ood(
            trans, rot, size,
            intrinsic, extrinsic,
            carla_data.bev_resolution,
            carla_data.bev_start_position,
            type='carla',
        )

        label[bev_ood == 1, :] = 0

        print(os.path.join(save_path + "bev_semantic", f'{index}.png'))
        cv2.imwrite(os.path.join(save_path + "bev_semantic", f'{index}.png'), cv2.cvtColor(label, cv2.COLOR_BGR2RGB))

        for i in range(6):
            sensor_name = cameras[i]
            image = Image.open(os.path.join(agent_path + sensor_name, f'{index}.png'))

            if i == cam:
                sc = 2
                image_r = F.interpolate(images[None, cam],
                                        scale_factor=sc, mode='bilinear', align_corners=False)
                mask_r = F.interpolate(torch.tensor(cam_ood[None, None]),
                                       scale_factor=sc, mode='bilinear', align_corners=False)

                result = pipe(
                    prompt=get_prompt(animals[a]),
                    image=image_r * 2 - 1,
                    mask_image=mask_r,
                    width=480 * sc, height=224 * sc,
                    output_type='np',
                ).images[0]

                cam_image = cv2.resize(result, (480, 224)) * 255
            else:
                cam_image = np.array(image)

            os.makedirs(os.path.join(save_path, sensor_name), exist_ok=True)
            cv2.imwrite(os.path.join(save_path + sensor_name, f'{index}.png'), cv2.cvtColor(cam_image, cv2.COLOR_BGR2RGB))


if __name__ == "__main__":
    generate()