import os
import time
import torch
from config import device
from utils import to_ssim, to_psnr, tensor2im, save_image
from gradient_sam import create_gradient_masks


def train(vis, train_loader, test_loader, model, cls_model, criterion, optimizer, scheduler, opts):
    tic = time.time()
    model.train()
    print('Training...')

    best_psnr = 0
    for idx, (snow_images, gt_desnow, gt_mask) in enumerate(train_loader):

        # ------------------- cuda -------------------
        snow_images = snow_images.to(device)                                  # [B, 3, 240, 240]
        gt_desnow = gt_desnow.to(device)                                      # [B, 3, 240, 240]
        gt_mask = gt_mask.to(device)                                          # [B, 1, 240, 240]

        # ------------------- make gradient masks -------------------
        grad_mask = create_gradient_masks(snow_images, cls_model)            # [B, 1, 240, 240]

        # ------------------- forward -------------------
        output = model(snow_images, grad_mask)  # output : (a, _, _)

        pred_desnow = output[0]
        pred_mask = output[1]
        pred_edge = output[2]

        # ------------------- backward -------------------
        loss, (l1, l2, l3, l4, l5) = criterion(pred_desnow, pred_mask, pred_edge, gt_desnow, gt_mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx > 50000 and idx % 10000 == 0:
            scheduler.step()
            print(scheduler.get_last_lr())

        # -------------------- print ------------------------
        toc = time.time()

        if idx % opts.vis_step == 0:
            print(str(idx) + '/' + str(len(train_loader)) + ', it takes {:.4f} sec'.format(toc - tic))

            # -------------------- loss plot ------------------------
            vis.line(X=torch.ones((1, 6)).cpu() * idx,  # step
                     Y=torch.Tensor([loss, l1, l2, l3, l4, l5]).unsqueeze(0).cpu(),
                     win='train_loss',
                     update='append',
                     opts=dict(xlabel='step',
                               ylabel='Loss',
                               title='training loss',
                               legend=['Total Loss', 'reconstruction loss', 'perceptual loss', 'edge_loss', 'snow_mask_loss', 'mask_edge_refine_loss']))

            # -------------------- vis images ------------------------
            # input image
            vis_batch = 0

            # input
            vis.image(img=snow_images[vis_batch].clamp(0, 1),
                      win='input',
                      opts=dict(title='input'))

            # gt_desnow
            vis.image(img=gt_desnow[vis_batch].clamp(0, 1),
                      win='gt_desnow',
                      opts=dict(title='gt_desnow',
                                width=240,
                                height=240))

            # pred_desnow
            vis.image(img=pred_desnow[vis_batch].clamp(0, 1),
                      win='pred_desnow',
                      opts=dict(title='pred_desnow',
                                width=240,
                                height=240))

            # gt_mask
            vis.image(img=gt_mask[vis_batch].squeeze().clamp(0, 1),
                      win='gt_mask',
                      opts=dict(title='gt_mask',
                                width=240,
                                height=240
                                ))

            # pred_mask
            vis.image(img=pred_mask[vis_batch].squeeze().clamp(0, 1),
                      win='pred_mask',
                      opts=dict(title='pred_mask',
                                width=240,
                                height=240
                                ))

            # grad_masks
            vis.image(img=grad_mask[vis_batch].squeeze().clamp(0, 1),
                      win='grad_masks',
                      opts=dict(title='grad_masks',
                                width=240,
                                height=240
                                ))

            # 200000 이상만 psnr 구한다.
            if idx > 200000 and idx % opts.vis_step == 0:

                psnr, ssim = val(idx, model, cls_model, test_loader)
                if best_psnr < psnr:
                    best_psnr = psnr
                    print("Best psnr : {:.4f}".format(best_psnr))
                    if not os.path.exists(opts.save_path):
                        os.mkdir(opts.save_path)
                    torch.save(model.state_dict(), os.path.join(opts.save_path, opts.save_file_name + '.{:.4f}.pth.tar'.format(best_psnr)))

                # -------------------- evaluation plot ------------------------
                vis.line(X=torch.ones((1, 2)).cpu() * idx,  # step
                         Y=torch.Tensor([psnr, ssim]).unsqueeze(0).cpu(),
                         win='val_psnr/ssim',
                         update='append',
                         opts=dict(xlabel='step',
                                   ylabel='value',
                                   title='val_psnr/ssim',
                                   legend=['psnr', 'ssim']))


def val(iter, model, cls_model, test_loader):
    tic = time.time()
    psnr_list = []
    ssim_list = []

    print("At {}-th iteration, validate psnr and ssim for {}'s data.".format(iter, len(test_loader)))
    for idx, (snow_images, gt_desnow, _) in enumerate(test_loader):

        # ------------------- cuda -------------------
        snow_images = snow_images.to(device)
        gt_desnow = gt_desnow.to(device)
        grad_masks = create_gradient_masks(snow_images, cls_model)

        with torch.no_grad():
            output = model.eval()(snow_images, grad_masks)  # output : (a, _, _)

        desnow = output[0]
        psnr_list.extend(to_psnr(desnow, gt_desnow))
        ssim_list.extend(to_ssim(desnow, gt_desnow))

    # -------------------- eval ------------------------
    avg_psnr = sum(psnr_list) / len(psnr_list)
    avg_ssim = sum(ssim_list) / len(ssim_list)
    toc = time.time()
    print('Result:', avg_psnr, avg_ssim, ', and it takes {:.4f} sec'.format(toc - tic))
    return avg_psnr, avg_ssim