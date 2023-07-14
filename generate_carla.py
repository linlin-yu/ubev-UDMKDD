from datasets.carla import *
from tools.utils import *
import random
from tqdm import tqdm

import torch.nn.functional as F
import matplotlib.pyplot as plt

from diffusers import StableDiffusionInpaintPipeline
from xformers.ops import MemoryEfficientAttentionFlashAttentionOp


def get_prompt(animal):
    return f"A {animal} standing on the road. The {animal} should be as large as possible. Make the {animal} photorealistic"


def generate():

    carla_data = CarlaDataset(data_path, False)

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16,
    )

    pipe.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
    pipe.vae.enable_xformers_memory_efficient_attention(attention_op=None)
    pipe = pipe.to(gpu)
    pipe.set_progress_bar_config(disable=True)

    animals = [
        "bear",
        "elephant",
        "horse",
        "deer"
    ]

    size = [
        [],
        [],
        [],
        [],
    ]

    cameras = [
        'left_front_camera',
        'front_camera',
        'right_front_camera',
        'left_back_camera',
        'back_camera',
        'right_back_camera'
    ]


    for i in tqdm(range(start, end)):
        images, intrinsics, extrinsics, labels, ood = carla_data[i]

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

        cv2.imwrite(os.path.join(save_path + "bev_semantic", f'{index}.png'), label)

        for i in range(6):
            sensor_name = cameras[i]
            image = Image.open(os.path.join(agent_path + sensor_name, f'{index}.png'))
            if i == cam:
                w, h = (960, 448)
                # w, h = (512, 240)

                image = image.resize((w, h))
                cam_ood = cv2.resize(cam_ood, (w, h))

                image = pipe(
                    prompt=get_prompt(animals[a]),
                    image=image,
                    mask_image=cam_ood,
                    width=w, height=h,
                ).images[0]
                image = image.resize((480, 224))

            os.makedirs(os.path.join(save_path, sensor_name), exist_ok=True)
            image.save(os.path.join(save_path + sensor_name, f'{index}.png'))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--start', required=True, type=int)
    parser.add_argument('-e', '--end', required=True, type=int)
    parser.add_argument('-g', '--gpu', required=True, type=int)
    parser.add_argument('-l', '--output', required=True, type=str)
    parser.add_argument('-d', '--datapath', required=True, type=str)
    args = parser.parse_args()

    start = args.start
    end = args.end
    gpu = args.gpu
    data_path = args.datapath
    save_dir = args.output

    generate()