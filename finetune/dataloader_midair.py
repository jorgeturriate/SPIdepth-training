# This file is mostly taken from BTS; author: Jin Han Lee, with only slight modifications

import os
import random

import numpy as np
import torch
import torch.utils.data.distributed
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import skimage.transform


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def preprocessing_transforms(mode):
    return transforms.Compose([
        ToTensor(mode=mode)
    ])


class DepthDataLoader(object):
    def __init__(self, args, mode):
        if mode == 'train':
            self.training_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            self.train_sampler = None
            self.data = DataLoader(self.training_samples, args.batch_size,
                                   shuffle=(self.train_sampler is None),
                                   num_workers=args.num_threads,
                                   pin_memory=True,
                                   sampler=self.train_sampler)

        elif mode == 'online_eval':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            self.eval_sampler = None
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=False,
                                   num_workers=1,
                                   pin_memory=False,
                                   sampler=self.eval_sampler)

        elif mode == 'test':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            self.data = DataLoader(self.testing_samples, 1, shuffle=False, num_workers=1)

        else:
            print('mode should be one of \'train, test, online_eval\'. Got {}'.format(mode))


def remove_leading_slash(s):
    if s[0] == '/' or s[0] == '\\':
        return s[1:]
    return s


class DataLoadPreprocess(Dataset):
    def __init__(self, args, mode, transform=None, is_for_online_eval=False):
        self.args = args
        if mode == 'online_eval':
            with open(args.filenames_file_eval, 'r') as f:
                self.filenames = f.readlines()
        else:
            with open(args.filenames_file, 'r') as f:
                self.filenames = f.readlines()

        self.mode = mode
        self.transform = transform
        self.to_tensor = ToTensor
        self.is_for_online_eval = is_for_online_eval
        self.full_res_shape = (1024, 1024)
        self.focal = 512.0  # MidAir approx focal length

    def __getitem__(self, idx):
        sample_path = self.filenames[idx]
        focal = self.focal

        if self.mode == 'train':
            image_path = os.path.join(self.args.data_path, remove_leading_slash(sample_path.split()[0]), f"{sample_path.split()[2]:06d}.JPEG" )
            depth_path = os.path.join(self.args.gt_path, remove_leading_slash(sample_path.split()[1]), f"{sample_path.split()[2]:06d}.PNG" )

            image = Image.open(image_path).convert('RGB')
            depth_gt = Image.open(depth_path)
            depth_gt = np.array(depth_gt, dtype=np.uint16)
            depth_float16 = depth_gt.view(dtype=np.float16)
            disparity = depth_float16.astype(np.float32)
            disparity[disparity == 0] = 0.01  # avoid division by 0
            depth_gt = 512. / disparity

            # Resize both to fixed 1024x1024 (MidAir resolution)
            image = image.resize(self.full_res_shape, Image.Resampling.BILINEAR)
            depth_gt = skimage.transform.resize(
                depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant'
            )

            image = np.asarray(image, dtype=np.float32) / 255.0
            depth_gt = np.expand_dims(depth_gt, axis=2)

            # image, depth_gt = self.random_crop(image, depth_gt, self.args.input_height, self.args.input_width)
            image = skimage.transform.resize(image, (self.args.input_height, self.args.input_width), order=1, preserve_range=True, mode='constant')
            depth_gt = skimage.transform.resize(depth_gt, (self.args.input_height, self.args.input_width), order=0, preserve_range=True, mode='constant')
            image, depth_gt = self.train_preprocess(image, depth_gt)
            sample = {'image': image, 'depth': depth_gt, 'focal': focal}

        else:
            if self.mode == 'online_eval':
                data_path = self.args.data_path_eval
            else:
                data_path = self.args.data_path

            image_path = os.path.join(data_path, remove_leading_slash(sample_path.split()[0]), f"{sample_path.split()[2]:06d}.JPEG")
            image = Image.open(image_path).convert('RGB')
            image = image.resize(self.full_res_shape, Image.Resampling.BILINEAR)
            image = np.asarray(image, dtype=np.float32) / 255.0

            if self.mode == 'online_eval':
                gt_path = self.args.gt_path_eval
                depth_path = os.path.join(gt_path, remove_leading_slash(sample_path.split()[1]), f"{sample_path.split()[2]:06d}.PNG")
                has_valid_depth = False
                try:
                    depth_gt = Image.open(depth_path)
                    has_valid_depth = True
                except IOError:
                    depth_gt = False
                    print('Missing gt, expected {}'.format(depth_path))
                    # print('Missing gt for {}'.format(image_path))

                if has_valid_depth:

                    depth_gt = np.array(depth_gt, dtype=np.uint16)
                    depth_float16 = depth_gt.view(dtype=np.float16)
                    disparity = depth_float16.astype(np.float32)
                    disparity[disparity == 0] = 0.01  # avoid division by 0
                    depth_gt = 512. / disparity
                    depth_gt = skimage.transform.resize(
                        depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant'
                    )
                    depth_gt = np.expand_dims(depth_gt, axis=2)

            
            if self.mode == 'online_eval':
                image = image.resize((self.args.input_width, self.args.input_height), Image.Resampling.BILINEAR)
                depth_gt = skimage.transform.resize(depth_gt, (self.args.input_height, self.args.input_width), order=0, preserve_range=True, mode='constant')

                sample = {'image': image, 'depth': depth_gt, 'focal': focal, 'has_valid_depth': has_valid_depth,
                          'image_path': sample_path.split()[0], 'depth_path': sample_path.split()[1]}
            else:
                sample = {'image': image, 'focal': focal}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def random_crop(self, img, depth, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]
        return img, depth

    def train_preprocess(self, image, depth_gt):
        # Random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()

        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)

        return image, depth_gt

    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug

    def __len__(self):
        return len(self.filenames)


class ToTensor(object):
    def __init__(self, mode):
        self.mode = mode
        # self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # for SQLdepth and LGM with mc pt_weights
        self.normalize = torch.nn.Identity() # for Zoedepth

    def __call__(self, sample):
        image, focal = sample['image'], sample['focal']
        image = self.to_tensor(image)
        image = self.normalize(image)

        if self.mode == 'test':
            return {'image': image, 'focal': focal}

        depth = sample['depth']
        if self.mode == 'train':
            depth = self.to_tensor(depth)
            return {'image': image, 'depth': depth, 'focal': focal}
        else:
            has_valid_depth = sample['has_valid_depth']
            return {'image': image, 'depth': depth, 'focal': focal, 'has_valid_depth': has_valid_depth,
                    'image_path': sample['image_path'], 'depth_path': sample['depth_path']}

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img

