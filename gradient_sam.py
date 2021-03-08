import torch
import torch.nn.functional as F
from config import device


def create_gradient_masks(input_img, cls_model):

    # input preprocessing
    input_img.requires_grad = True
    batch_size = input_img.size(0)
    width = input_img.size(2)
    height = input_img.size(3)
    cls_model.eval()
    batch_masks = []
    pred_cls, k = cls_model(input_img)

    for b in range(batch_size):
        target_labels = 1  # snow part
        grads = torch.autograd.grad(pred_cls[b][target_labels], k, retain_graph=True)[0]
        one_grad = grads[b]
        a_k = torch.mean(one_grad, dim=(1, 2), keepdim=True)
        mask = torch.sum(a_k * k[b], dim=0).to(device)
        out = mask

        # normalization
        out = (out + torch.abs(out)) / 2
        if out.sum() == 0:  # no division by zero
            out = out
        else:
            out = out / torch.max(out)  # 0 ~ 1 scaling

        # up-sampling
        out = F.upsample_bilinear(out.unsqueeze(0).unsqueeze(0), [width, height])  # 4D로 바꿈

        batch_masks.append(out)
    batch_masks = torch.stack(batch_masks, dim=0).squeeze(1)  # [B, 1, 1, 240, 240] --> [B, 1, 240, 240]
    return batch_masks
