

import numpy as np

from tqdm import tqdm

import glob

import torch
import models

from utils import PSNR,_getHW,_getYdata,to_pixel_samples
UPSCALE = 2     # upscaling factor
TEST_DIR = './TestSet'      # Test images
EXP_NAME = "ArbSR_0423"

global model


def init(upscale, QP):
    global model
    if(QP>205):
        model_spec = torch.load(r"./transfer/{}\NAF_light_fold9_dim4_sin_1616_bilinearadd_qp210\epoch-best.pth".format(EXP_NAME))['model']
    elif QP>175:
        model_spec = torch.load(r"./transfer/{}\NAF_light_fold9_dim4_sin_1616_bilinearadd_qp185\epoch-best.pth".format(EXP_NAME))['model']
    elif QP>145:
        model_spec = torch.load(r"./transfer/{}\NAF_light_fold9_dim4_sin_1616_bilinearadd_qp160\epoch-best.pth".format(EXP_NAME))['model']
    else:
        model_spec = torch.load(r"./transfer/{}\NAF_light_fold9_dim4_sin_1616_bilinearadd_qp135\epoch-best.pth".format(EXP_NAME))['model']

    model = models.make(model_spec, load_sd=True).cuda()

def wrapper(im,gt_H,gt_W):

    crop_lr = im
    hr_coord = to_pixel_samples(gt_H,gt_W)

    cell = torch.ones_like(hr_coord)
    cell[:, 0] *= 2 / int(gt_H)
    cell[:, 1] *= 2 / int(gt_W)

    return {
        'inp': crop_lr[None, :, :],
        'coord': hr_coord[None, :, :],
        'cell': cell[None, :, :],
    }

def superres(src,H,W):
    img_lr = np.array(src).astype(np.float32)
    # save_img("img_lr.jpg", img_lr)
    img_lr = img_lr/255
    img_lr = torch.from_numpy(img_lr[None, :, :]).cuda()
    c,h, w = img_lr.shape

    inp_sub = torch.FloatTensor([0.5]).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor([0.5]).view(1, -1, 1, 1).cuda()
    gt_sub = torch.FloatTensor([0.5]).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor([0.5]).view(1, 1, -1).cuda()

    batch = wrapper(img_lr, H, W)

    for k, v in batch.items():
        batch[k] = v.cuda()

    inp = (batch['inp'] - inp_sub) / inp_div

    with torch.no_grad():
        pred = model(inp, batch['coord'], batch['cell'])

    pred = pred * gt_div + gt_sub
    pred.clamp_(0, 1)
    # reshape
    pred = pred.reshape([1, int(H), int(W), 1])
    pred = pred[0,:,:,0].cpu().numpy()
    pred = np.round(pred * 255).astype('int')
    # save_img("tmp_yuv.jpg",pred)
    pred = pred.tolist()

    return pred

if __name__ == '__main__':
    global model
    QP=135

    init(None,QP)


    # Test LR images
    files_lr = glob.glob(TEST_DIR + '/div2k_val_LR_AVM/3by2/qp{}/*.yuv'.format(QP))
    files_lr.sort()

    # Test GT images
    files_gt = glob.glob(TEST_DIR + '/label/*.yuv')
    files_gt.sort()

    psnrs = []
    for ti, fn in enumerate(tqdm(files_gt)):
        # Load LR image
        W, H = _getHW(files_lr[ti], -1)
        tmp = _getYdata(files_lr[ti],[W, H],norm=False)

        # Process LR image
        W, H = _getHW(files_gt[ti], -4)
        pred  = superres(tmp,H,W)
        # Load GT image
        W, H = _getHW(files_gt[ti], -4)
        tmp = _getYdata(files_gt[ti],[W, H],norm=False)
        img_gt = np.asarray(tmp).astype(np.float32)  # HxWxC


        psnr = PSNR(pred, img_gt)

        print(files_gt[ti], "\n psnr:", psnr)
        psnrs.append(psnr)




    print('AVG PSNR: {}'.format(np.mean(np.asarray(psnrs))))