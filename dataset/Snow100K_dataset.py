import os
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader, RandomSampler
import torchvision.transforms.functional as F
import wget
import tarfile
import random
from torchvision import transforms
import numpy as np
import torch


def bar_custom(current, total, width=0):
    progress = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
    return progress


def download_snow100k_dataset(root="D:\data\Snow_100k", remove_compressed_file=True):
    """

    """
    train_data_url = "https://desnownet.s3.amazonaws.com/dataset_synthetic/train/Snow100K-training.tar.gz"
    test_data_url = "https://desnownet.s3.amazonaws.com/dataset_synthetic/test/Snow100K-testset.tar.gz"
    real_data_url = "https://desnownet.s3.amazonaws.com/realistic_image/realistic.tar.gz"

    os.makedirs(root, exist_ok=True)

    if (os.path.exists(os.path.join(root, 'Snow100K-training')) and
            os.path.exists(os.path.join(root, 'Snow100K-testset')) and
            os.path.exists(os.path.join(root, 'realistic'))):
        print("Already exist!")
        return

    print('Download...')
    wget.download(url=train_data_url, out=root, bar=bar_custom)
    print('')
    wget.download(url=test_data_url, out=root, bar=bar_custom)
    print('')
    wget.download(url=real_data_url, out=root, bar=bar_custom)
    print('')
    # each dataset has 7.8GB / 7.8GB / 67MB

    print('Extract...')
    with tarfile.open(os.path.join(root, 'Snow100K-training.tar.gz')) as tar:
        tar.extractall(os.path.join(root, 'Snow100K-training'))
    with tarfile.open(os.path.join(root, 'Snow100K-testset.tar.gz')) as tar:
        tar.extractall(os.path.join(root, 'Snow100K-testset'))
    with tarfile.open(os.path.join(root, 'realistic.tar.gz')) as tar:
        tar.extractall(os.path.join(root, 'realistic'))

    # remove tars
    if remove_compressed_file:
        root_zip_list = glob.glob(os.path.join(root, '*.gz'))  # in root_dir remove *.zip
        for root_zip in root_zip_list:
            os.remove(root_zip)
        print("Remove *.tar.gz(s)")
    print("Done!")


class Snow100K_Dataset(Dataset):

    def __init__(self, root, split, iteration=None, num_sample=None, snow_size='L', download=True):
        """
        Snow100K_Dataset for testing
        :param root: Snow 100K root {train, test}
        :param split: which process do you want
        :param iteration: batch * step_size
        :param snow_size: snow size consist of 'L', 'M', 'S'
        :param download: if download snow100k datasets
        """
        # -------------------------- download --------------------------
        self.download = download
        self.iteration = iteration
        if self.download:
            download_snow100k_dataset(root=root)

        self.root = root
        self.split = split
        assert split in ['train', 'test']
        assert snow_size in ['S', 'M', 'L']

        if self.split == 'test':
            self.snow_paths = os.path.join(self.root, 'Snow100K-testset', 'media', 'jdway', 'GameSSD', 'overlapping', 'test', 'Snow100K-{}'.format(snow_size), 'synthetic')
            self.gt_paths = os.path.join(self.root, 'Snow100K-testset', 'media', 'jdway', 'GameSSD', 'overlapping', 'test', 'Snow100K-{}'.format(snow_size), 'gt')
            self.mask_paths = os.path.join(self.root, 'Snow100K-testset', 'media', 'jdway', 'GameSSD', 'overlapping', 'test', 'Snow100K-{}'.format(snow_size), 'mask')

        if self.split == 'train':
            self.snow_paths = os.path.join(self.root, 'Snow100K-training', 'all', 'synthetic')
            self.gt_paths = os.path.join(self.root, 'Snow100K-training', 'all', 'gt')
            self.mask_paths = os.path.join(self.root, 'Snow100K-training', 'all', 'mask')

        self.snow_paths = sorted(glob.glob(os.path.join(self.snow_paths, '*.jpg')))
        self.gt_paths = sorted(glob.glob(os.path.join(self.gt_paths, '*.jpg')))
        self.mask_paths = sorted(glob.glob(os.path.join(self.mask_paths, '*.jpg')))

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop((240, 240), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
        ])

        if num_sample is None:
            num_sample = len(self.snow_paths)
        print("num_sample :", num_sample)

        # 중복없이
        random.seed(1)
        self.snow_paths = random.sample(self.snow_paths, num_sample)
        random.seed(1)
        self.gt_paths = random.sample(self.gt_paths, num_sample)
        random.seed(1)
        self.mask_paths = random.sample(self.mask_paths, num_sample)

    def __len__(self):
        if self.split == 'test':
            return len(self.snow_paths)
        return self.iteration

    def __getitem__(self, index):

        if self.split == 'train':
            index = random.randint(0, len(self.snow_paths) - 1)
            # print(index)

        img_name = os.path.basename(self.snow_paths[index])[:-4]  # .jpg 삭제
        snow_image = Image.open(self.snow_paths[index]).convert('RGB')
        gt_desnow = Image.open(self.gt_paths[index]).convert('RGB')
        gt_mask = Image.open(self.mask_paths[index])

        if self.split == 'train':
            # 같은 RandomResizedCrop 등을 연산하려고
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
    NUM_STEPS = 200000
    batch_size = 2
    num_sample = 50000
    # dataset = Snow100K_Dataset(root='D:\data\Snow_100k', split='train', iteration=NUM_STEPS * batch_size, num_sample=num_sample)
    dataset = Snow100K_Dataset(root='/home/cvmlserver4/Sungmin/data/Snow_100k', split='train', iteration=NUM_STEPS * batch_size, num_sample=num_sample)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=2,
                             pin_memory=True)

    for data in data_loader:
        print("data_loader : ", len(data_loader))


