import os
import cv2
import numpy as np
import torch
import torchvision
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from albumentations import (
    Compose, HorizontalFlip, RandomRotate90, RandomBrightnessContrast,
    ElasticTransform, GridDistortion, RandomCrop, Normalize,Resize
)
from torch.nn import functional as F
from albumentations.pytorch import ToTensorV2

readvdnames = lambda x: open(x).read().rstrip().split('\n')


################################# DEFINE DATASET #################################
class TinySegDataset(Dataset):
    def __init__(self, db_root="TinySeg", img_size=128, phase='train', aug_mode='none'):
        self.classes = ['person', 'bird', 'car', 'cat', 'plane']
        self.seg_ids = [1, 2, 3, 4, 5]  # 原始标签为1~5

        templ_image = db_root + "/JPEGImages/{}.jpg"
        templ_mask = db_root + "/Annotations/{}.png"
        ids = readvdnames(db_root + "/ImageSets/" + phase + ".txt")

        self.samples = [
            [templ_image.format(i), templ_mask.format(i)]
            for i in ids
        ]
        self.phase = phase
        self.db_root = db_root
        self.img_size = img_size
        self.aug_mode = aug_mode

        # 基础增强（水平翻转、旋转、亮度对比度）
        base_aug = [
            HorizontalFlip(p=0.5),
            RandomRotate90(p=0.5),
            RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        ]

        # 完整增强（弹性变换、网格畸变、随机裁剪）
        full_aug = base_aug + [
            ElasticTransform(alpha=0.5, sigma=50, alpha_affine=30, p=0.3),
            GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
            RandomCrop(width=256, height=256, p=0.3),
        ]

        # 后处理（尺寸调整、归一化到-1~1、Tensor转换）
        post_process = [
            Resize(img_size, img_size),
            ToTensorV2()  # 转换为Tensor但不归一化
        ]

        # 组合增强策略
        if aug_mode == 'none':
            self.transform = Compose(post_process)
        elif aug_mode == 'base':
            self.transform = Compose(base_aug + post_process)
        elif aug_mode == 'full':
            self.transform = Compose(full_aug + post_process)
        else:
            raise ValueError(f"Invalid aug_mode: {aug_mode}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.phase == 'train':
            return self.get_train_item(idx)
        else:
            return self.get_test_item(idx)

    def get_train_item(self, idx):
        img_path, mask_path = self.samples[idx]

        # 读取图像（BGR转RGB）
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 读取调色板掩膜
        mask_pil = Image.open(mask_path).convert('P')
        mask = np.array(mask_pil, dtype=np.uint8)

        # 应用数据增强
        transformed = self.transform(image=image, mask=mask)

        # 提取Tensor并手动归一化到-1~1
        image = transformed['image'].float()  # [0,255] -> [0.0,1.0]
        image = (image / 127.5) - 1.0  # 归一化到-1~1

        mask = transformed['mask'].long()

        cv2.imwrite("test.png", np.concatenate([(image[0]+1)*127.5, mask*255], axis=0))
        return image, mask

    def get_test_item(self, idx):
        img_path, mask_path = self.samples[idx]

        # 读取图像（BGR转RGB）
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 读取调色板掩膜
        mask_pil = Image.open(mask_path).convert('P')
        mask = np.array(mask_pil, dtype=np.uint8)

        # 应用数据增强
        transformed = self.transform(image=image, mask=mask)

        # 提取Tensor并手动归一化到-1~1
        image = transformed['image'].float()  # [0,255] -> [0.0,1.0]
        image = (image / 127.5) - 1.0  # 归一化到-1~1

        mask = transformed['mask'].long()

        return image, mask

# class TinySegDataset(Dataset):
#     def __init__(self, root_dir, split='train', aug_mode='none', img_size=128):
#         self.root = os.path.join(root_dir, split)
#         self.images = sorted([
#             os.path.join(self.root, 'images', f)
#             for f in os.listdir(os.path.join(self.root, 'images'))
#         ])
#         self.masks = sorted([
#             os.path.join(self.root, 'masks', f)
#             for f in os.listdir(os.path.join(self.root, 'masks'))
#         ])
#         self.aug_mode = aug_mode
#         self.img_size = img_size
#
#         # 定义基础增强（水平翻转、旋转、亮度对比度）
#         base_aug = [
#             HorizontalFlip(p=0.5),
#             RandomRotate90( p=0.5),
#             RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3)
#         ]
#
#         # 定义完整增强（添加弹性变换、网格畸变、随机裁剪）
#         full_aug = base_aug + [
#             ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
#             GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
#             RandomCrop(width=96, height=96, p=0.5)
#         ]
#
#         # 统一的后处理（尺寸调整、归一化、Tensor转换）
#         post_process = [
#             Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
#             Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#             ToTensorV2()
#         ]
#
#         # 组合增强策略
#         if aug_mode == 'none':
#             self.transform = Compose(post_process)
#         elif aug_mode == 'base':
#             self.transform = Compose(base_aug + post_process)
#         elif aug_mode == 'full':
#             self.transform = Compose(full_aug + post_process)
#         else:
#             raise ValueError(f"Invalid aug_mode: {aug_mode}")
#
#     def __len__(self):
#         return len(self.images)
#
#     def __getitem__(self, idx):
#         # 读取图像和掩码
#         image = cv2.imread(self.images[idx], cv2.IMREAD_COLOR)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)
#
#         # 应用数据增强
#         transformed = self.transform(image=image, mask=mask)
#         image, mask = transformed['image'], transformed['mask']
#
#         # 确保掩码为整数类型
#         mask = mask.long()
#         return image, mask

def get_dataloader(root_dir, img_size,phase='train',batch_size=32, aug_mode='none'):
    # dataset = TinySegDataset(root_dir, phase='train', aug_mode=aug_mode)
    # loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    # return loader
    dataset = TinySegDataset(
        db_root=root_dir,
        img_size=img_size,
        phase=phase,
        aug_mode=aug_mode
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(phase == 'train'),
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )