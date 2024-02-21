import os.path as osp
import pickle
import random
from typing import Dict

import numpy as np
import torch
import torch.utils.data

from geotransformer.utils.pointcloud import (
    random_sample_rotation,
    random_sample_rotation_v2,
    get_transform_from_rotation_translation,
)
from geotransformer.utils.registration import get_correspondences

import matplotlib.image as image
import cv2

def adjust_intrinsic(intrinsic, intrinsic_image_dim, image_dim):

    if intrinsic_image_dim == image_dim:
        return intrinsic

    intrinsic_return = np.copy(intrinsic)

    height_after = image_dim[1]
    height_before = intrinsic_image_dim[1]
    height_ratio = height_after / height_before

    width_after = image_dim[0]
    width_before = intrinsic_image_dim[0]
    width_ratio = width_after / width_before

    if width_ratio >= height_ratio:
        resize_height = height_after
        resize_width = height_ratio * width_before
    else:
        resize_width = width_after
        resize_height = width_ratio * height_before

    intrinsic_return[0,0] *= float(resize_width)/float(width_before)
    intrinsic_return[1,1] *= float(resize_height)/float(height_before)
    # account for cropping/padding here
    intrinsic_return[0,2] *= float(resize_width-1)/float(width_before-1)
    intrinsic_return[1,2] *= float(resize_height-1)/float(height_before-1)
    
    return intrinsic_return

def process_image(image, aim_H=480, aim_W=640, mode="resize", clip_mode="center"):

    H, W, C = np.array(image).shape

    '''
        H x W
        min:(1513,2141)
        max:(339,396)
    '''

    if (H == aim_H and W == aim_W):
        return np.array(image)

    if (mode == "resize"):
        # dsize = （W，H）
        image = np.asarray(
            cv2.resize(
                image,
                dsize=(aim_W, aim_H),
                interpolation=cv2.INTER_LINEAR
            ),
            dtype=np.float32
        )


    elif (mode == "clip"):

        while (H < aim_H or W < aim_W):
            image = cv2.pyrUp(src=image)
            H, W, C = np.array(image).shape

        if (H > aim_H * 2 and W > aim_W * 2):
            image = cv2.pyrDown(src=image)
            H, W, C = np.array(image).shape

        if (clip_mode == "center"):
            H_top = int((H - aim_H) / 2)
            W_left = int((W - aim_W) / 2)
            image = image[H_top:H_top + aim_H, W_left:W_left + aim_W]
        elif (clip_mode == "normal"):
            image = image[0:aim_H, 0:aim_W]
        elif (clip_mode == "random"):
            H_top = int(np.random.random() * (H - aim_H))
            W_left = int(np.random.random() * (W - aim_W))
            image = image[H_top:H_top + aim_H, W_left:W_left + aim_W]

    elif (mode == "padding"):
        # (C,H,W)
        image = np.transpose(image, (2, 0, 1))

        if(aim_H < H and aim_W < W):
            padding_H = aim_H - H
            padding_W = aim_W - W

            padding_H0 = np.zeros((C, padding_H, W))
            padding_W0 = np.zeros((C, aim_H, padding_W))

            image = np.concatenate([image, padding_H0], axis=1)
            image = np.concatenate([image, padding_W0], axis=2)
        elif(aim_H < H):
            image = image[:,0:aim_H,:]

            padding_W = aim_W - W
            padding_W0 = np.zeros((C, aim_H, padding_W))

            image = np.concatenate([image,padding_W0],axis=2)

        elif(aim_W < W):
            image = image[:,:,0:aim_W]

            padding_H = aim_H - H
            padding_H0 = np.zeros((C, padding_H, W))

            image = np.concatenate([image, padding_H0], axis=1)
        else:

            image = image[:,0:aim_H,0:aim_W]


        image = np.transpose(image,(1,2,0))

    return image

class ThreeDMatchPairDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_num,
        dataset_root,
        subset,
        point_limit=None,
        use_augmentation=False,
        augmentation_noise=0.005,
        augmentation_rotation=1,
        overlap_threshold=None,
        return_corr_indices=False,
        matching_radius=None,
        rotated=False,
    ):
        super(ThreeDMatchPairDataset, self).__init__()

        self.image_num = image_num

        self.dataset_root = dataset_root
        self.metadata_root = osp.join(self.dataset_root, 'metadata')
        self.data_root = osp.join(self.dataset_root, 'data')

        self.subset = subset
        self.point_limit = point_limit
        self.overlap_threshold = overlap_threshold
        self.rotated = rotated

        self.return_corr_indices = return_corr_indices
        self.matching_radius = matching_radius
        if self.return_corr_indices and self.matching_radius is None:
            raise ValueError('"matching_radius" is None but "return_corr_indices" is set.')

        self.use_augmentation = use_augmentation
        self.aug_noise = augmentation_noise
        self.aug_rotation = augmentation_rotation

        with open(osp.join(self.metadata_root, f'{subset}.pkl'), 'rb') as f:
            self.metadata_list = pickle.load(f)
            if self.overlap_threshold is not None:
                self.metadata_list = [x for x in self.metadata_list if x['overlap'] > self.overlap_threshold]

    def __len__(self):
        return len(self.metadata_list)


    def read_int(self, scene):
        if self.subset == 'train' or self.subset == 'val':
            intrinsics = np.loadtxt(osp.join(self.data_root, 'train', scene, 'camera-intrinsics.txt'))
        else:
            intrinsics = np.loadtxt(osp.join(self.data_root, 'test', scene, 'camera-intrinsics.txt'))
        big_size, image_size = [640, 480], [160, 120]
        intrinsics = adjust_intrinsic(intrinsics, big_size, image_size)
        if intrinsics.shape[0] == 3:
            res = np.eye(4)
            res[:3, :3] = intrinsics
            intrinsics = res
        intrinsics = torch.from_numpy(intrinsics).float()
        return intrinsics
    
    
    def read_image(self, filename):
        posename = filename.replace('.png', '_pose.txt')
        if(not osp.exists(filename)):
            filename = filename.replace('.png', '.jpg')
            if(not osp.exists(filename)):
                print(filename, 'not existed!')
                exit()
        img = image.imread(filename)
        if(img.shape[0] != 240 or img.shape[1] != 320):
            img = process_image(image=img, aim_H=240, aim_W=320)
        img = np.transpose(img, axes=(2, 0, 1))
        pose = np.loadtxt(posename)
        return img, pose
    
    def _load_point_cloud_and_image(self, file_name, datadict):
        scene = datadict['scene_name']
        intrinsics = self.read_int(scene)
        
        point_path = osp.join(self.data_root, file_name)
        points = torch.load(point_path)
        # NOTE: setting "point_limit" with "num_workers" > 1 will cause nondeterminism.
        if self.point_limit is not None and points.shape[0] > self.point_limit:
            indices = np.random.permutation(points.shape[0])[: self.point_limit]
            points = points[indices]

        image_path = point_path
        image_path = image_path.replace(".pth", "_0.png")
        if self.image_num == 1:
            if(not osp.exists(image_path)):
                image_path = image_path.replace(".png",".jpg")
                if(not osp.exists(image_path)):
                    print(image_path, 'not existed!')
                    exit()
            img = image.imread(image_path)
            if(img.shape[0] != 240 or img.shape[1] != 320):
                img = process_image(image = img, aim_H = 240, aim_W = 320)
            img = np.transpose(img,axes=(2,0,1))
        elif self.image_num == 2:
            img1, pose1 = self.read_image(image_path)
            image_path = image_path.replace('_0.png', '_1.png')
            img2, pose2 = self.read_image(image_path)
            img = {'img1':img1, 'img2':img2, 'pose1':pose1, 'pose2':pose2}

        return points, img, intrinsics
    
    def _augment_point_cloud(self, ref_points, src_points, rotation, translation):
        r"""Augment point clouds.

        ref_points = src_points @ rotation.T + translation

        1. Random rotation to one point cloud.
        2. Random noise.
        """
        aug_rotation = random_sample_rotation(self.aug_rotation)
        ref_rotation, src_rotation = np.eye(4), np.eye(4)
        if random.random() > 0.5:
            ref_points = np.matmul(ref_points, aug_rotation.T)
            rotation = np.matmul(aug_rotation, rotation)
            translation = np.matmul(aug_rotation, translation)
            ref_rotation[:3, :3] = aug_rotation.T
        else:
            src_points = np.matmul(src_points, aug_rotation.T)
            rotation = np.matmul(rotation, aug_rotation.T)
            src_rotation[:3, :3] = aug_rotation.T

        ref_points += (np.random.rand(ref_points.shape[0], 3) - 0.5) * self.aug_noise
        src_points += (np.random.rand(src_points.shape[0], 3) - 0.5) * self.aug_noise

        return ref_points, src_points, rotation, translation, ref_rotation, src_rotation

    def __getitem__(self, index):
        data_dict = {}

        # metadata
        metadata: Dict = self.metadata_list[index]
        data_dict['scene_name'] = metadata['scene_name']
        data_dict['ref_frame'] = metadata['frag_id0']
        data_dict['src_frame'] = metadata['frag_id1']
        data_dict['overlap'] = metadata['overlap']

        # get transformation
        rotation = metadata['rotation']
        translation = metadata['translation']

        # get point cloud and image
        ref_points, ref_image, ref_intrinsics = self._load_point_cloud_and_image(metadata['pcd0'], data_dict)
        src_points, src_image, src_intrinsics = self._load_point_cloud_and_image(metadata['pcd1'], data_dict)

        # augmentation
        ref_world2camera_1, src_world2camera_1 = np.eye(4), np.eye(4)
        if self.use_augmentation:
            ref_points, src_points, rotation, translation, ref_world2camera_1, src_world2camera_1 = self._augment_point_cloud(
                ref_points, src_points, rotation, translation
            )

        if self.rotated:
            ref_rotation = random_sample_rotation_v2()
            ref_points = np.matmul(ref_points, ref_rotation.T)
            rotation = np.matmul(ref_rotation, rotation)
            translation = np.matmul(ref_rotation, translation)

            src_rotation = random_sample_rotation_v2()
            src_points = np.matmul(src_points, src_rotation.T)
            rotation = np.matmul(rotation, src_rotation.T)

        transform = get_transform_from_rotation_translation(rotation, translation)
        # get correspondences
        if self.return_corr_indices:
            corr_indices = get_correspondences(ref_points, src_points, transform, self.matching_radius)
            data_dict['corr_indices'] = corr_indices
        
        data_dict['ref_points'] = ref_points.astype(np.float32)
        data_dict['src_points'] = src_points.astype(np.float32)
        data_dict['ref_feats'] = np.ones((ref_points.shape[0], 1), dtype=np.float32)
        data_dict['src_feats'] = np.ones((src_points.shape[0], 1), dtype=np.float32)
        data_dict['transform'] = transform.astype(np.float32)
        data_dict['ref_intrinsics'] = ref_intrinsics
        data_dict['src_intrinsics'] = src_intrinsics

        if self.image_num == 1:
            data_dict['ref_image'] = ref_image
            data_dict['src_image'] = src_image
            data_dict['ref_rotation'] = ref_world2camera_1
            data_dict['src_rotation'] = src_world2camera_1
        elif self.image_num == 2:
            data_dict['ref_image_1'] = ref_image['img1']
            data_dict['ref_image_2'] = ref_image['img2']
            data_dict['ref_world2camera_1'] =  ref_world2camera_1.astype(np.float32)
            ref_pose2_rev = np.linalg.inv(ref_image['pose2'])
            data_dict['ref_world2camera_2'] = (ref_pose2_rev @ ref_image['pose1'] @ ref_world2camera_1).astype(np.float32)
            
            data_dict['src_image_1'] = src_image['img1']
            data_dict['src_image_2'] = src_image['img2']
            data_dict['src_world2camera_1'] =  src_world2camera_1.astype(np.float32)
            src_pose2_rev = np.linalg.inv(src_image['pose2'])
            data_dict['src_world2camera_2'] = (src_pose2_rev @ src_image['pose1'] @ src_world2camera_1).astype(np.float32)
        
 
        return data_dict
