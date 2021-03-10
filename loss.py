import torch
import torch.nn as nn
from torchvision.models import vgg19, vgg19_bn, vgg16
from config import device
import torch.nn.functional as F


class Get_gradient(nn.Module):
    def __init__(self):
        super(Get_gradient, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def forward(self, x):

        if x.size(1) == 1:
            x0 = x[:, 0]
            x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=1)
            x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=1)
            x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
            x = x0
        else:
            x0 = x[:, 0]
            x1 = x[:, 1]
            x2 = x[:, 2]
            x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=1)
            x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=1)

            x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding=1)
            x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding=1)

            x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding=1)
            x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding=1)

            x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
            x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
            x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)
            x = torch.cat([x0, x1, x2], dim=1)
        return x


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
        mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        return (batch - mean) / std

    def forward(self, out_images, target_images):

        loss = self.l1_loss(
            self.loss_network(self.normalize_batch(out_images)),
            self.loss_network(self.normalize_batch(target_images))
        )

        return loss


class DualGradLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.l1_loss = nn.L1Loss().to(device)
        self.content_loss = ContentLoss().to(device)
        self.get_edge = Get_gradient().to(device)

    def forward(self, pred_desnow, pred_mask, pred_edge, gt_desnow, gt_mask):

        # get_edge
        gt_edge = self.get_edge(gt_mask)

        edge_of_gt_desnow = self.get_edge(gt_desnow)
        edge_of_pred_desnow = self.get_edge(pred_desnow)

        # between desnow
        l1 = reconstruction_loss = self.l1_loss(pred_desnow, gt_desnow)                 # desnow 끼리
        l2 = perceptual_loss = self.content_loss(pred_desnow, gt_desnow)                # desnow 끼리

        # between edge of desnow
        l3 = edge_loss = self.l1_loss(edge_of_pred_desnow, edge_of_gt_desnow)           # desnow의 edge 끼리

        # between masks
        l4 = snow_mask_loss = self.l1_loss(pred_mask, gt_mask)                          # mask 끼리

        # between edge
        l5 = mask_edge_refine_loss = self.l1_loss(pred_edge, gt_edge)                   # edge 끼리

        # l2 *= 0.1
        loss = l1 + l2 + l3 + l4 + l5

        return loss, (l1, l2, l3, l4, l5)



