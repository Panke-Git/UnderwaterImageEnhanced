"""
    @Project: UnderwaterImageEnhanced
    @Author: Panke
    @FileName: dataset.py
    @Time: 2025/5/20 00:20
    @Email: None
"""
import os
import albumentations as A
import numpy as np
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset
import random

from tools.split_data import is_image_file


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])


class DataReader(Dataset):
    def __init__(self, img_dir, input='input', target='GT', model='train', ori=False, img_options=None):
        super(DataReader, self).__init__()

        input_files = sorted(os.listdir(os.path.join(img_dir, input)))
        target_files = sorted(os.listdir(os.path.join(img_dir, target)))

        self.input_filenames = [os.path.join(img_dir, input, x) for x in input_files if is_image_file(x)]
        self.target_filenames = [os.path.join(img_dir, target, x) for x in target_files if is_image_file(x)]
        self.model = model
        self.img_options = img_options
        self.sizex = len(self.input_filenames)

        if self.model == 'train':
            self.transform = A.Compose([
                A.OneOf([
                    A.HorizontalFlip(p=1.0),
                    A.VerticalFlip(p=1.0),
                ], p=0.3),
                A.RandomRotate90(p=0.3),
                A.Rotate(p=0.3),
                A.Transpose(p=0.3),
                A.RandomResizedCrop((img_options['h'], img_options['w']), )
            ],
                additional_targets={
                    'target': 'image',
                })

            self.degrade = A.Compose([
                A.NoOp()
            ])

        else:
            if ori:
                self.transform = A.Compose([
                    A.NoOp(),
                ],
                    additional_targets={
                        'target': 'image',
                    })
            else:
                self.transform = A.Compose([
                    A.Resize(img_options['h'], img_options['w']),
                ], additional_targets={
                    'target': 'image',
                })
            self.degrade = A.Compose([
                A.NoOp(),
            ])

    def mixup(self, inp_img, tar_img, mode='mixup'):
        mixup_index_ = random.randint(0, self.sizex - 1)

        _, transformed = self.load(mixup_index_)

        alpha = 0.2
        lam = np.random.beta(alpha, alpha)

        mixup_inp_img = F.to_tensor(self.degrade(image=transformed['image'])['image'])
        mixup_tar_img = F.to_tensor(transformed['target'])

        if mode == 'mixup':
            inp_img = lam * inp_img + (1 - lam) * mixup_inp_img
            tar_img = lam * tar_img + (1 - lam) * mixup_tar_img
        else:
            img_h, img_w = self.img_options['h'], self.img_options['w']

            cx = np.random.uniform(0, img_w)
            cy = np.random.uniform(0, img_h)

            w = img_w * np.sqrt(1 - lam)
            h = img_h * np.sqrt(1 - lam)

            x0 = int(np.round(max(cx - w / 2, 0)))
            x1 = int(np.round(min(cx + w / 2, img_w)))
            y0 = int(np.round(max(cy - h / 2, 0)))
            y1 = int(np.round(min(cy + h / 2, img_h)))

            inp_img[:, y0:y1, x0:x1] = mixup_inp_img[:, y0:y1, x0:x1]
            tar_img[:, y0:y1, x0:x1] = mixup_tar_img[:, y0:y1, x0:x1]

        return inp_img, tar_img

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex

        tar_path, transformed = self.load(index_)

        if self.mode == 'train':
            inp_img = F.to_tensor(self.degrade(image=transformed['image'])['image'])
        else:
            inp_img = F.to_tensor(transformed['image'])
        tar_img = F.to_tensor(transformed['target'])

        filename = os.path.basename(tar_path)

        return inp_img, tar_img, filename

    def get_labels(self):
        labels = []
        for filename in self.inp_filenames:
            label = None
            if 'uieb' in filename:
                label = 0
            elif 'euvp' in filename:
                label = 1
            elif 'lsui' in filename:
                label = 2

            labels.append(label)

        return labels

    def load(self, index_):
        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = Image.open(inp_path).convert('RGB')
        tar_img = Image.open(tar_path).convert('RGB')

        inp_img = np.array(inp_img)
        tar_img = np.array(tar_img)

        transformed = self.transform(image=inp_img, target=tar_img)

        return tar_path, transformed
