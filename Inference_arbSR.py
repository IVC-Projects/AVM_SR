
from PIL import Image
import numpy as np
from os import mkdir
from os.path import isdir
from tqdm import tqdm
import cv2
import glob
# from basicsr_utils import tensor2img
# from basicsr_metrics import calculate_psnr,calculate_ssim
import torch
import models
from Train_SPLUT_M import VERSION,EXP_NAME
from utils import calc_psnr,PSNR, GeneratorEnqueuer, DirectoryIterator_DIV2K, _load_img_array, _rgb2ycbcr,_getHW,_getYdata,to_pixel_samples

UPSCALE = 2     # upscaling factor
TEST_DIR = 'F:\JJ\DATASET\AV1_DATA\AVM_SUPERRES_DATA'      # Test images



global model



# # Test LR images
# files_lr = glob.glob(TEST_DIR + '/div2k_val_LR_AVM/x{}/qp{}/*.yuv'.format(UPSCALE,QP))
# files_lr.sort()
#
# # Test GT images
# files_gt = glob.glob(TEST_DIR + '/div2k_val_HR/label/*.yuv')
# files_gt.sort()

L = 16

def save_img(filename, rec):
    cv2.imwrite(filename, rec)

def init(upscale, QP):
    global model
    if(QP>205):
        model_spec = torch.load(r"save\NAF_light_fold9_dim4_sin_1616_bilinearadd_qp210\epoch-best.pth")['model']
    elif QP>175:
        model_spec = torch.load(r"save\NAF_light_fold9_dim4_sin_1616_bilinearadd_qp185\epoch-best.pth")['model']
    elif QP>145:
        model_spec = torch.load(r"save\NAF_light_fold9_dim4_sin_1616_bilinearadd_qp160\epoch-best.pth")['model']
    else:
        model_spec = torch.load(r"save\NAF_light_fold9_dim4_sin_1616_bilinearadd_qp135\epoch-best.pth")['model']

    model = models.make(model_spec, load_sd=True).cuda()

def wrapper(im,gt_H,gt_W):
    # print(gt_H,gt_W)
        # [1,540,960] chw
        # [1,1080,1920] chw
        # s = np.random.choice((5/4,6/4,7/4,8/4))  # denote 5/4,6/4,7/4,8/4
        # s_img_set = {5/4:im5,6/4:im6,7/4:im7,8/4:im8}
        # crop_lr = s_img_set

        # img = s_img_set[s]
        # w_lr = self.inp_size
        # print(img.shape)
        # x0 = random.randint(0, img.shape[-2] - w_lr+1)
        # y0 = random.randint(0, img.shape[-1] - w_lr+1)
        # print(img.shape)
        # print(gt.shape,s)
        # print(x0,x0 + w_lr,y0, y0 + w_lr,int(x0*s),int((x0 + w_lr)*s),int(y0*s),int((y0 + w_lr)*s))

    crop_lr = im


    hr_coord = to_pixel_samples(gt_H,gt_W)
    # hr_coord, hr_rgb = to_pixel_samples(crop_hr)

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
    files_gt = glob.glob(TEST_DIR + '/div2k_val_HR/label/*.yuv')
    files_gt.sort()

    psnrs = []
    for ti, fn in enumerate(tqdm(files_gt)):
        # Load LR image
        W, H = _getHW(files_lr[ti], -1)
        tmp = _getYdata(files_lr[ti],[W, H],norm=True)
        # img_lr = np.asarray(tmp).astype(np.float32)  # HxW
        # img_lr = torch.from_numpy(img_lr[None, :, :]).cuda()
        W, H = _getHW(files_gt[ti], -2)
        pred  = superres(tmp,H,W)
        # Load GT image
        W, H = _getHW(files_gt[ti], -2)
        tmp = _getYdata(files_gt[ti],[W, H],norm=False)
        img_gt = np.asarray(tmp).astype(np.float32)  # HxWxC
        # img_gt = torch.from_numpy(img_gt[None,:, :,None]).cuda()


        # inp_sub = torch.FloatTensor([0.5]).view(1, -1, 1, 1).cuda()
        # inp_div = torch.FloatTensor([0.5]).view(1, -1, 1, 1).cuda()
        # gt_sub = torch.FloatTensor([0.5]).view(1, 1, -1).cuda()
        # gt_div = torch.FloatTensor([0.5]).view(1, 1, -1).cuda()


        # batch = wrapper(img_lr, H,W)
        #
        # for k, v in batch.items():
        #     batch[k] = v.cuda()
        #
        # inp = (batch['inp'] - inp_sub) / inp_div
        #
        # with torch.no_grad():
        #     pred = model(inp, batch['coord'], batch['cell'])
        #
        # pred = pred * gt_div + gt_sub
        # pred.clamp_(0, 1)

        psnr = PSNR(pred, img_gt)

        print(files_gt[ti], "\n psnr:", psnr)
        psnrs.append(psnr)




    print('AVG PSNR: {}'.format(np.mean(np.asarray(psnrs))))