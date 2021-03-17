import sys

import torch
import torch.optim as optim
import visdom
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from config import parse, device, device_ids
from dataset.Snow100K_dataset import Snow100K_Dataset
from dataset.srrs_dataset import SRRS_Dataset
from loss import DualGradLoss
from model import Dual_Grad_Desnow_Net, VGG
from train import train


def main():

    # argparser
    opts = parse(sys.argv[1:])
    print(opts)

    # visdom
    vis = visdom.Visdom(port=opts.port)

    NUM_STEPS = opts.iteration    # 200000
    batch_size = opts.batch_size  # 4
    num_sample = 50000

    # dataset
    train_set = None
    test_set = None

    if opts.data_type == 'snow100k':
        train_set = Snow100K_Dataset(root=opts.root, split='train', iteration=NUM_STEPS * batch_size, num_sample=num_sample)
        test_set = Snow100K_Dataset(root=opts.root, split='test', num_sample=1000)

    elif opts.data_type == 'srrs':
        train_set = SRRS_Dataset(root=opts.root, split='train', iteration=NUM_STEPS * batch_size)
        test_set = SRRS_Dataset(root=opts.root, split='test')

    train_loader = DataLoader(dataset=train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=2,
                              pin_memory=True)

    test_loader = DataLoader(dataset=test_set,
                             batch_size=1,
                             shuffle=True,
                             num_workers=0,
                             pin_memory=True)

    model = Dual_Grad_Desnow_Net().to(device)
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    cls_model = VGG().to(device)

    if opts.data_type == 'snow100k':
        # snow100k cls weight
        cls_state_dict = torch.load('./models/snow100k_classification_vgg_16.pth.tar', map_location=device)
        cls_model.load_state_dict(cls_state_dict, strict=True)

    elif opts.data_type == 'srrs':
        # srrs cls weight
        cls_state_dict = torch.load('./models/srrs_classification_vgg_16.pth.tar', map_location=device)
        cls_model.load_state_dict(cls_state_dict, strict=True)

    cls_model.eval()
    criterion = DualGradLoss()

    betas = (0.9, 0.999)
    optimizer = optim.Adam(model.parameters(), lr=opts.lr, betas=betas, weight_decay=opts.weight_decay)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    train(vis=vis,
          train_loader=train_loader,
          test_loader=test_loader,
          model=model,
          cls_model=cls_model,
          criterion=criterion,
          optimizer=optimizer,
          scheduler=scheduler,
          opts=opts)

    return


if __name__ == '__main__':
    main()
