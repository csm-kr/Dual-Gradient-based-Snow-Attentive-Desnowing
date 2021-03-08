import pytorch_ssim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, vgg19_bn, vgg16
from config import device
from math import log10
from PIL import Image
import numpy as np


# --- Perceptual loss network  --- #
class ContentLoss(nn.Module):

    def __init__(self):
        super(ContentLoss, self).__init__()

        vgg = vgg19(pretrained=True).to(device)
        self.loss_network = nn.Sequential(*list(vgg.features)[:22]).eval()
        for param in self.loss_network.parameters():
            param.requires_grad = False

        self.l1_loss = nn.L1Loss()

    def normalize_batch(self, batch):
        # Normalize batch using ImageNet mean and std
        mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        return (batch - mean) / std

    def forward(self, out_images, target_images):

        loss = self.l1_loss(
            self.loss_network(self.normalize_batch(out_images)),
            self.loss_network(self.normalize_batch(target_images))
        )

        return loss


def to_psnr(output, gt):
    output = torch.clamp(output, 0, 1)
    mse = F.mse_loss(output, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]

    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list


def to_ssim(output, gt):
    output = torch.clamp(output, 0, 1)
    ssim_list = [pytorch_ssim.ssim(output, gt).item()]
    return ssim_list


def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].detach().cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255
    image_numpy = np.clip(image_numpy, 0, 255)

    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    if image_numpy.shape[2] == 1:
        image_numpy = np.reshape(image_numpy, (image_numpy.shape[0], image_numpy.shape[1]))
        image_pil = Image.fromarray(image_numpy, 'L')
    else:
        image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)