from nuscenes.utils.geometry_utils import (box_in_image,
                                           view_points)

from datasets.nuscenes import *
from tools.geometry import *

from diffusers import StableDiffusionInpaintPipeline
from xformers.ops import MemoryEfficientAttentionFlashAttentionOp


class NuScenesDatasetOOD(NuScenesDataset):
    def __init__(self, *args, **kwargs):
        super(NuScenesDatasetOOD, self).__init__(*args, **kwargs)

        # pipe = StableDiffusionInpaintPipeline.from_pretrained(
        #     "stabilityai/stable-diffusion-2-inpainting",
        #     torch_dtype=torch.float16,
        # )
        
        # pipe.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
        # pipe.vae.enable_xformers_memory_efficient_attention(attention_op=None)

    def __getitem__(self, index):
        rec = self.ixes[index]

        images, intrinsics, extrinsics = self.get_input_data(rec)
        labels = self.get_label(rec)

        self.nusc.get('ego_pose',
                      self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])

        ood = np.zeros(self.bev_dimension[:2])

        trans = [10., 0., 0.]
        size = [4., 4., 4.]
        rot = [0., 0., 0., 1.]

        box = Box(trans, size, Quaternion(rot))

        pts = box.bottom_corners()[:2].T

        pts = np.round(
            (pts - self.bev_start_position[:2] + self.bev_resolution[:2] / 2.0) / self.bev_resolution[:2]
        ).astype(np.int32)

        pts[:, [1, 0]] = pts[:, [0, 1]]
        cv2.fillPoly(ood, [pts], 1.0)

        cam_ood = np.zeros((len(self.cameras), 270, 480))

        for cam in range(len(self.cameras)):
            crec = self.nusc.get('sample_data', rec['data'][self.cameras[cam]])
            cs_record = self.nusc.get('calibrated_sensor', crec['calibrated_sensor_token'])
            cam_intrinsic = np.array(cs_record['camera_intrinsic'])

            cb = box.copy()

            cb.translate(-np.array(cs_record['translation']))
            cb.rotate(Quaternion(cs_record['rotation']).inverse)

            if box_in_image(cb, cam_intrinsic, (900, 1600)):
                corners = view_points(cb.corners(), cam_intrinsic, normalize=True)[:2, :]
                corners = np.int32(corners * .3).T

                xmin, ymin, xmax, ymax = bounding_box(corners)

                xmin, ymin = max(xmin, 0), max(ymin, 0)
                xmax, ymax = min(xmax, 480), max(ymax, 270)

                cam_ood[cam, ymin:ymax, xmin:xmax] = 1
                print(cam)

        cam_ood = cam_ood[:, :224, :]

        return images, intrinsics, extrinsics, labels, ood, cam_ood


def compile_data(version, dataroot, batch_size=8, num_workers=16):
    nusc, dataroot = get_nusc(version, dataroot)

    train_data = NuScenesDatasetOOD(nusc, True)
    val_data = NuScenesDatasetOOD(nusc, False)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    return train_loader, val_loader
