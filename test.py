import os
import time
import torch
from config import device, device_ids
from torch.utils.data import DataLoader
from dataset.Snow100K_dataset import Snow100K_Dataset
from dataset.srrs_dataset import SRRS_Dataset
from model import Dual_Grad_Desnow_Net, VGG
from utils import to_ssim, to_psnr, tensor2im, save_image
from gradient_sam import create_gradient_masks


def test(data_loader, model, cls_model, opts):
    """
    test dual gradient desnowing
    :param data_loader: loader of test datset
    :param model: model for desnowing
    :param cls_model: model for classification
    :return: -
    """

    # ------------------- load -------------------
    tic = time.time()
    print('Test...')

    if opts.data_type == 'snow100k':

    #     # snow100k weight
    #     if opts.eval_type == 's':
    #         state_dict = torch.load(
    #             os.path.join(opts.save_path, opts.save_file_name) + '.{}.{:.4f}.pth.tar'.format(opts.eval_type, opts.ssim),
    #             map_location=device)
    #     elif opts.eval_type == 'p':
    #         state_dict = torch.load(
    #             os.path.join(opts.save_path, opts.save_file_name) + '.{}.{:.4f}.pth.tar'.format(opts.eval_type, opts.psnr),
    #             map_location=device)

        state_dict = torch.load('./models/snow100k_model_params.pth.tar', map_location=device)
        cls_state_dict = torch.load('./models/snow100k_classification_vgg_16.pth.tar', map_location=device)

    elif opts.data_type == 'srrs':

        # # srrs weight
        # if opts.eval_type == 's':
        #     state_dict = torch.load(
        #         os.path.join(opts.save_path, opts.save_file_name) + '.{}.{}.pth.tar'.format(opts.eval_type, opts.ssim),
        #         map_location=device)
        # elif opts.eval_type == 'p':
        #     state_dict = torch.load(
        #         os.path.join(opts.save_path, opts.save_file_name) + '.{}.{}.pth.tar'.format(opts.eval_type, opts.psnr),
        #         map_location=device)

        state_dict = torch.load('./models/srrs_model_params.pth.tar', map_location=device)
        cls_state_dict = torch.load('./models/srrs_classification_vgg_16.pth.tar', map_location=device)

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    cls_model.load_state_dict(cls_state_dict, strict=True)
    cls_model.eval()

    psnr_list = []
    ssim_list = []

    for idx, (snow_images, gt_images, img_name) in enumerate(data_loader):

        # ------------------- cuda -------------------
        snow_images = snow_images.to(device)
        gt_images = gt_images.to(device)
        grad_masks = create_gradient_masks(snow_images, cls_model)

        with torch.no_grad():
            output = model.eval()(snow_images, grad_masks)  # output : (a, _, _)

        desnow = output[0]
        psnr_list.extend(to_psnr(desnow, gt_images))
        ssim_list.extend(to_ssim(desnow, gt_images))

        # -------------------- save ------------------------
        save = False
        if save:
            if not os.path.isdir('./test'):
                os.mkdir('./test')
            desnow = tensor2im(desnow)
            save_image(desnow, './test/' + 'desnow_' + img_name[0] + '.png')  # for img_name is tuple

        # -------------------- print ------------------------
        toc = time.time()
        if idx % 10 == 0:
            print(str(idx) + '/' + str(len(data_loader)) + ' done, it takes {:.4f} sec'.format(toc - tic))

    # -------------------- eval ------------------------
    avg_psnr = sum(psnr_list) / len(psnr_list)
    avg_ssim = sum(ssim_list) / len(ssim_list)
    print('Result:', avg_psnr, avg_ssim)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--data_type', type=str, default='snow100k', help='choose snow100k or srrs')
    # parser.add_argument('--root', type=str, default='D:\data\Snow_100k')
    # parser.add_argument('--root', type=str, default='D:\data\SRRS')
    parser.add_argument('--root', type=str, default='/home/cvmlserver5/Sungmin/data/Snow_100k')
    parser.add_argument('--save_path', type=str, default='./saves')
    parser.add_argument('--save_file_name', type=str, default='DualGradDeSnow2')       # FIXME
    parser.add_argument('--eval_type', type=str, default='p')                         # FIXME
    parser.add_argument('--psnr', type=float, default=29.0635)                        # FIXME
    parser.add_argument('--ssim', type=float, default=0.8959)                         # FIXME
    test_opts = parser.parse_args()
    print(test_opts)

    if test_opts.data_type == 'srrs':

        test_set = SRRS_Dataset(root=test_opts.root, split='test')
        test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)

    elif test_opts.data_type == 'snow100k':

        test_set = Snow100K_Dataset(root=test_opts.root, split='test', snow_size='S')  # FIXME - snow size
        test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)

    # model
    model = Dual_Grad_Desnow_Net().to(device)
    # model = torch.nn.DataParallel(model, device_ids=device_ids)
    cls_model = VGG().to(device)

    # test
    test(data_loader=test_loader,
         model=model,
         cls_model=cls_model,
         opts=test_opts)






