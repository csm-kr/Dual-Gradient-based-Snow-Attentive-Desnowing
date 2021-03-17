import os
import random
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader


class SRRS_Dataset(Dataset):

    def __init__(self, root="D:\data\SRRS", split='train', iteration=None):

        self.iteration = iteration
        self.root = root
        self.split = split

        self.root = os.path.join(root, "SRRS_15004")
        self.syn_paths = os.path.join(self.root, "Syn")
        self.snow_gt_paths = os.path.join(self.root, "gt_new")
        self.mask_gt_paths = os.path.join(self.root, "Combine Snow")

        self.syn_names = sorted(os.listdir(self.syn_paths))
        self.snow_gt_names = sorted(os.listdir(self.snow_gt_paths))
        self.mask_gt_names = sorted(os.listdir(self.mask_gt_paths))

        assert split in ['train', 'test']

        # get the 2500 samples
        if split == 'train':
            random.seed(1454)
            self.snow_names = random.sample(self.syn_names, 3500)
            random.seed(0)
            self.snow_names_for_test = random.sample(self.snow_names, 1000)
            set1 = set(self.snow_names)
            set2 = set(self.snow_names_for_test)
            self.snow_names = sorted(list(set1.difference(set2)))
            self.mask_gt_names = self.snow_gt_names = [snow_name.replace('.tif', '.jpg') for snow_name in self.snow_names]

        elif split == 'test':
            random.seed(1454)
            self.snow_names = random.sample(self.syn_names, 3500)
            random.seed(0)
            self.snow_names_for_test = random.sample(self.snow_names, 1000)
            self.snow_names = sorted(self.snow_names_for_test)
            self.mask_gt_names = self.snow_gt_names = [snow_name.replace('.tif', '.jpg') for snow_name in self.snow_names]

        # self.num_samples = num_samples
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop((240, 240), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
        ])

    def __len__(self):
        if self.split == 'test':
            return len(self.snow_names)
        return self.iteration

    def __getitem__(self, index):

        if self.split == 'train':
            index = random.randint(0, len(self.snow_names) - 1)

        # Snow ---------------------------------------------------------------------------------------------------------
        snow_name, snow_gt_name, mask_gt_name = self.snow_names[index], self.snow_gt_names[index], self.mask_gt_names[index]

        snow_image = Image.open(os.path.join(self.syn_paths, snow_name))
        gt_desnow = Image.open(os.path.join(self.snow_gt_paths, snow_gt_name))
        gt_mask = Image.open(os.path.join(self.mask_gt_paths, mask_gt_name))

        # --------------------------------------------------------------------------------------------------------------
        if self.split == 'train':

            seed = np.random.randint(2021)
            random.seed(seed)
            torch.manual_seed(seed)
            snow_image = self.transform(snow_image)

            random.seed(seed)
            torch.manual_seed(seed)
            gt_desnow = self.transform(gt_desnow)

            random.seed(seed)
            torch.manual_seed(seed)
            gt_mask = self.transform(gt_mask)

        else:
            snow_image = snow_image
            gt_desnow = gt_desnow
            gt_mask = gt_mask

        snow_image = F.to_tensor(snow_image)
        gt_desnow = F.to_tensor(gt_desnow)
        gt_mask = F.to_tensor(F.to_grayscale(gt_mask))

        visualize = False
        if visualize:
            import matplotlib.pyplot as plt
            # x
            image = snow_image
            img_vis = np.array(image.permute(1, 2, 0), np.float32)  # C, W, H
            img_vis = np.clip(img_vis, 0, 1)
            plt.figure('input')
            plt.imshow(img_vis)

            # desnow
            image = gt_desnow
            img_vis = np.array(image.permute(1, 2, 0), np.float32)  # C, W, H
            img_vis = np.clip(img_vis, 0, 1)
            plt.figure('desnow')
            plt.imshow(img_vis)

            # mask
            image = gt_mask
            img_vis = np.array(image.permute(1, 2, 0), np.float32)  # C, W, H
            img_vis = np.concatenate([img_vis, img_vis, img_vis], axis=2)  # C, W, H
            img_vis = np.clip(img_vis, 0, 1)
            plt.figure('mask')
            plt.imshow(img_vis)
            plt.show()

        return snow_image, gt_desnow, gt_mask


if __name__ == '__main__':
    # dataset = SRRS_Dataset(split='train')
    # for k in dataset:
    #     print(k[2].size())

    NUM_STEPS = 200000
    batch_size = 2

    # dataset = SRRS_Dataset(root='D:\data\SRRS', split='train', iteration=NUM_STEPS * batch_size)
    dataset = SRRS_Dataset(root='/home/cvmlserver4/Sungmin/data/Snow/SRRS', split='train', iteration=NUM_STEPS * batch_size)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=2,
                             pin_memory=True)

    for data in data_loader:
        print("data_loader : ", len(data_loader))
        print(data)