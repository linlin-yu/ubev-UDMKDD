import os
import warnings

import torch
import math
import torchvision
from PIL import Image

import numpy as np
from tools.geometry import *


class CarlaDataset(torch.utils.data.Dataset):
    def __init__(self, dataroot, is_train):
        self.is_train = is_train

        self.dataroot = dataroot

        self.mode = 'train' if self.is_train else 'val'

        self.vehicles = len(os.listdir(os.path.join(self.data_path, 'agents')))
        self.ticks = len(os.listdir(os.path.join(self.data_path, 'agents/0/back_camera')))

        self.to_tensor = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        bev_resolution, bev_start_position, bev_dimension = calculate_birds_eye_view_parameters(
            [-50.0, 50.0, 0.5], [-50.0, 50.0, 0.5], [-10.0, 10.0, 20.0]
        )

        self.bev_resolution, self.bev_start_position, self.bev_dimension = (
            bev_resolution.numpy(), bev_start_position.numpy(), bev_dimension.numpy()
        )

        self.cameras = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

    def get_input_data(self, index, agent_path):
        images = []
        intrinsics = []
        extrinsics = []

        for sensor_name, sensor_info in self.sensors_info['sensors'].items():
            if sensor_info["sensor_type"] == "sensor.camera.rgb" and sensor_name != "birds_view_camera":
                image = Image.open(os.path.join(agent_path + sensor_name, f'{index}.png'))

                intrinsic = sensor_info["transform"]["intrinsic"]
                extrinsic = sensor_info["transform"]["extrinsic"]

                normalized_image = self.to_tensor(image)

                images.append(normalized_image)
                intrinsics.append(intrinsic)
                extrinsics.append(inverse_extrinsics(extrinsic))

        images, intrinsics, extrinsics = (torch.stack(images, dim=0),
                                          torch.stack(intrinsics, dim=0),
                                          torch.stack(extrinsics, dim=0))

        return images, intrinsics, extrinsics

    def get_label(self, index, agent_path):
        label_r = Image.open(os.path.join(agent_path + "birds_view_semantic_camera", f'{index}.png'))
        label = np.array(label_r)
        label_r.close()

        empty = np.ones((200, 200))

        road = mask(label, (128, 64, 128))
        lane = mask(label, (157, 234, 50))
        vehicles = mask(label, (0, 0, 142))

        empty[vehicles == 1] = 0
        empty[road == 1] = 0
        empty[lane == 1] = 0
        label = np.stack((vehicles, road, lane, empty))

        return torch.tensor(label)

    def __len__(self):
        return self.ticks * self.vehicles

    def __getitem__(self, index):
        agent_number = math.floor(index / self.ticks)
        agent_path = os.path.join(self.data_path, f"agents/{agent_number}/")
        index = (index + self.offset) % self.ticks

        images, intrinsics, extrinsics = self.get_input_data(index, agent_path)
        labels = self.get_label(index, agent_path)

        return images, intrinsics, extrinsics, labels


def compile_data(version, dataroot, batch_size=8, num_workers=16):
    train_data = CarlaDataset(dataroot, True)
    val_data = CarlaDataset(dataroot, False)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    return train_loader, val_loader