
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

from utils  import PSNR, GeneratorEnqueuer, DirectoryIterator_DIV2K, _load_img_array, _rgb2ycbcr,_getHW,_getYdata
LR_G = 1e-3         # Learning rate for the generator
QQ=2**4
OUT_NUM=4

VERSION = "SPLUT_M_{}_{}".format(LR_G,OUT_NUM)

UPSCALE = 2     # upscaling factor
TEST_DIR = 'F:\JJ\DATASET\AV1_DATA\AVM_SUPERRES_DATA'      # Test images

# TEST_DIR = './test_dataset/Set14'
# TEST_DIR ='./test_dataset/BSDS100'
# TEST_DIR ='./test_dataset/urban100'
# TEST_DIR ='./test_dataset/manga109'

global LUTA1_122, LUTA2_221, LUTA2_212, LUTA3_221, LUTA3_212
global LUTB1_122, LUTB2_221, LUTB2_212, LUTB3_221, LUTB3_212


# # Test LR images
# files_lr = glob.glob(TEST_DIR + '/div2k_val_LR_AVM/x{}/qp{}/*.yuv'.format(UPSCALE,QP))
# files_lr.sort()
#
# # Test GT images
# files_gt = glob.glob(TEST_DIR + '/div2k_val_HR/label/*.yuv')
# files_gt.sort()

L = 16

def LUT1_122(weight, img_in):
    C, H, W = img_in.shape
    img_in = img_in.reshape(C, 1, H, W)

    img_a1 = img_in[:,:, 0:H-1, 0:W-1]
    img_b1 = img_in[:,:, 0:H-1, 1:W  ]
    img_c1 = img_in[:,:, 1:H  , 0:W-1]
    img_d1 = img_in[:,:, 1:H  , 1:W  ]

    out = weight[ img_a1.flatten().astype(np.int_)*L*L*L + img_b1.flatten().astype(np.int_)*L*L + img_c1.flatten().astype(np.int_)*L + img_d1.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], -1))   
    out = np.transpose(out, (0, 1, 4, 2, 3)).reshape((img_a1.shape[0], -1,img_a1.shape[2],img_a1.shape[3]))
    return out

def LUT23(weight, img_in,a_range,ker,nlut):
    img_in = np.clip(img_in, -a_range, a_range-0.01)
    B ,IC, H, W = img_in.shape
    if ker=='k221':
        img_a1 = img_in[:,0:IC-1, 0:H-1,:]+L/2
        img_b1 = img_in[:,0:IC-1, 1:H  ,:]+L/2
        img_c1 = img_in[:,1:IC  , 0:H-1,:]+L/2
        img_d1 = img_in[:,1:IC  , 1:H  ,:]+L/2
    else:
        img_a1 = img_in[:,0:IC-1, :,0:W-1]+L/2
        img_b1 = img_in[:,0:IC-1, :,1:W  ]+L/2
        img_c1 = img_in[:,1:IC  , :,0:W-1]+L/2
        img_d1 = img_in[:,1:IC  , :,1:W  ]+L/2
    out = weight[ img_a1.flatten().astype(np.int_)*L*L*L + img_b1.flatten().astype(np.int_)*L*L + img_c1.flatten().astype(np.int_)*L + img_d1.flatten().astype(np.int_) ]
    if nlut==2:
        out=out.reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], -1))   
        out = np.transpose(out, (0, 1, 4, 2, 3)).reshape((img_a1.shape[0], -1,img_a1.shape[2],img_a1.shape[3]))
    else:
        out=out.reshape((img_a1.shape[0], img_a1.shape[2], img_a1.shape[3],-1))   
        out=np.transpose(out, (0, 3, 1, 2))
    return out

def channel_shuffle(x, groups):

    batchsize, num_channels, height, width = x.shape
    channels_per_group = num_channels // groups
    # num_channels = groups * channels_per_group

    # grouping, 通道分组
    # b, num_channels, h, w =======>  b, groups, channels_per_group, h, w
    x = np.reshape(x,[batchsize, groups, channels_per_group, height, width])
    # channel shuffle, 通道洗牌
    x = x.transpose(0,2,1,3,4)
    # x.shape=(batchsize, channels_per_group, groups, height, width)
    # flatten
    x = np.reshape(x, [batchsize, -1, height, width])

    return x

def save_img(filename, rec):

    cv2.imwrite(filename, rec)

def init(upscale, QP):
    EXP_NAME = "YUV{}_SRNet_v2_1227".format(str(QP))
    global LUTA1_122, LUTA2_221, LUTA2_212, LUTA3_221, LUTA3_212
    global LUTB1_122, LUTB2_221, LUTB2_212, LUTB3_221, LUTB3_212
    # Load LUT
    LUTA2_221 = np.load(
        "./transfer/{}_{}/LUT{}_K{}_{}_Model_S_A.npy".format(VERSION, EXP_NAME, 2, 221, str(QP))).astype(np.float32)
    LUTA2_212 = np.load(
        "./transfer/{}_{}/LUT{}_K{}_{}_Model_S_A.npy".format(VERSION, EXP_NAME, 2, 212, str(QP))).astype(np.float32)
    LUTA3_221 = np.load(
        "./transfer/{}_{}/LUT{}_K{}_{}_Model_S_A.npy".format(VERSION, EXP_NAME, 3, 221, str(QP))).astype(np.float32)
    LUTA3_212 = np.load(
        "./transfer/{}_{}/LUT{}_K{}_{}_Model_S_A.npy".format(VERSION, EXP_NAME, 3, 212, str(QP))).astype(np.float32)
    LUTA1_122 = np.load(
        "./transfer/{}_{}/LUT{}_K{}_{}_Model_S_A.npy".format(VERSION, EXP_NAME, 1, 122, str(QP))).astype(np.float32)

    LUTB1_122 = np.load(
        "./transfer/{}_{}/LUT{}_K{}_{}_Model_S_B.npy".format(VERSION, EXP_NAME, 1, 122, str(QP))).astype(np.float32)
    LUTB2_221 = np.load(
        "./transfer/{}_{}/LUT{}_K{}_{}_Model_S_B.npy".format(VERSION, EXP_NAME, 2, 221, str(QP))).astype(np.float32)
    LUTB2_212 = np.load(
        "./transfer/{}_{}/LUT{}_K{}_{}_Model_S_B.npy".format(VERSION, EXP_NAME, 2, 212, str(QP))).astype(np.float32)
    LUTB3_221 = np.load(
        "./transfer/{}_{}/LUT{}_K{}_{}_Model_S_B.npy".format(VERSION, EXP_NAME, 3, 221, str(QP))).astype(np.float32)
    LUTB3_212 = np.load(
        "./transfer/{}_{}/LUT{}_K{}_{}_Model_S_B.npy".format(VERSION, EXP_NAME, 3, 212, str(QP))).astype(np.float32)

def superres(src,upscale):
    global LUTA1_122, LUTA2_221, LUTA2_212, LUTA3_221, LUTA3_212
    global LUTB1_122, LUTB2_221, LUTB2_212, LUTB3_221, LUTB3_212
    img_lr = np.array(src).astype(np.float32)
    img_lr = img_lr[:, :,None]
    h, w, c = img_lr.shape
    # print("img_lr.shape",h,w,c)

    img_in = np.pad(img_lr, ((0, 1), (0, 1), (0, 0)), mode='reflect').transpose((2, 0, 1))
    img_in_A255 = img_in // L
    img_in_B255 = img_in % L
    img_in_A = img_in_A255 / L
    img_in_B = img_in_B255 / L

    # A
    x_layer1 = LUT1_122(LUTA1_122, img_in_A255) + img_in_A[:, 0:h, 0:w].reshape((1, 1, h, w))
    # v2
    x_layer1 = channel_shuffle(x_layer1, 2)
    # v2
    x2_in1 = np.pad(x_layer1[:, 0:2, :, :], ((0, 0), (0, 0), (0, 1), (0, 0)), mode='reflect') + np.pad(x_layer1[:, 2:4, :, :], ((0, 0), (0, 0), (1, 0), (0, 0)), mode='reflect')
    x2_in2 = np.pad(x_layer1[:, 4:6, :, :], ((0, 0), (0, 0), (0, 0), (0, 1)), mode='reflect') + np.pad(x_layer1[:, 6:, :, :], ((0, 0), (0, 0), (0, 0), (1, 0)), mode='reflect')
    x_layer2 = (LUT23(LUTA2_221, x2_in1, 8.0, 'k221', 2) + LUT23(LUTA2_212, x2_in2, 8.0, 'k212', 2)) / 2.0 + x_layer1
    # v2
    x_layer2 = channel_shuffle(x_layer2, 2)
    # v2
    x3_in1 = np.pad(x_layer2[:, 0:2, :, :], ((0, 0), (0, 0), (0, 1), (0, 0)), mode='reflect') + np.pad(x_layer2[:, 2:4, :, :], ((0, 0), (0, 0), (1, 0), (0, 0)), mode='reflect')
    x3_in2 = np.pad(x_layer2[:, 4:6, :, :], ((0, 0), (0, 0), (0, 0), (0, 1)), mode='reflect') + np.pad(x_layer2[:, 6:, :, :], ((0, 0), (0, 0), (0, 0), (1, 0)), mode='reflect')
    img_out = (LUT23(LUTA3_221, x3_in1, 8.0, 'k221', 3) + LUT23(LUTA3_212, x3_in2, 8.0, 'k212', 3)) / 2.0 + img_in_A[:,0:h,0:w].reshape((1, 1, h, w))
    # v2
    img_out = channel_shuffle(img_out, 2)
    # v2
    img_out = img_out.reshape((1, UPSCALE, UPSCALE, h, w))
    img_out = np.transpose(img_out, (0, 3, 1, 4, 2)).reshape((1, UPSCALE * h, UPSCALE * w))
    img_out_A = img_out.transpose((1, 2, 0))
    # B
    x_layer1 = LUT1_122(LUTB1_122, img_in_B255) + img_in_B[:, 0:h, 0:w].reshape((1, 1, h, w))
    # v2
    x_layer1 = channel_shuffle(x_layer1, 2)
    # v2
    x2_in1 = np.pad(x_layer1[:, 0:2, :, :], ((0, 0), (0, 0), (0, 1), (0, 0)), mode='reflect') + np.pad(x_layer1[:, 2:4, :, :], ((0, 0), (0, 0), (1, 0), (0, 0)), mode='reflect')
    x2_in2 = np.pad(x_layer1[:, 4:6, :, :], ((0, 0), (0, 0), (0, 0), (0, 1)), mode='reflect') + np.pad(x_layer1[:, 6:, :, :], ((0, 0), (0, 0), (0, 0), (1, 0)), mode='reflect')
    x_layer2 = (LUT23(LUTB2_221, x2_in1, 8.0, 'k221', 2) + LUT23(LUTB2_212, x2_in2, 8.0, 'k212', 2)) / 2.0 + x_layer1
    # v2
    x_layer2 = channel_shuffle(x_layer2, 2)
    # v2
    x3_in1 = np.pad(x_layer2[:, 0:2, :, :], ((0, 0), (0, 0), (0, 1), (0, 0)), mode='reflect') + np.pad(x_layer2[:, 2:4, :, :], ((0, 0), (0, 0), (1, 0), (0, 0)), mode='reflect')
    x3_in2 = np.pad(x_layer2[:, 4:6, :, :], ((0, 0), (0, 0), (0, 0), (0, 1)), mode='reflect') + np.pad(x_layer2[:, 6:, :, :], ((0, 0), (0, 0), (0, 0), (1, 0)), mode='reflect')
    img_out = (LUT23(LUTB3_221, x3_in1, 8.0, 'k221', 3) + LUT23(LUTB3_212, x3_in2, 8.0, 'k212', 3)) / 2.0 + img_in_B[:,0:h,0:w].reshape((1, 1, h, w))
    # v2
    img_out = channel_shuffle(img_out, 2)
    # v2
    img_out = img_out.reshape((1, UPSCALE, UPSCALE, h, w))
    img_out = np.transpose(img_out, (0, 3, 1, 4, 2)).reshape((1, UPSCALE * h, UPSCALE * w))
    img_out_B = img_out.transpose((1, 2, 0))

    img_out = img_out_A + img_out_B
    img_out = np.round(np.clip(img_out, 0, 1) * 255).astype(np.uint8)


    # print("img_out.shape", img_out.shape)
    # print("-------------------")

    img_out = img_out[:, :, 0]

    # rec = np.asarray(img_out, dtype='float32')
    # im = Image.fromarray(rec)
    # im.show()
    # save_img(rec)

    rec = np.around(img_out)
    rec = rec.astype('int')
    rec = rec.tolist()
    return rec

if __name__ == '__main__':
    QP=210
    init(2, QP)

    # Test LR images
    files_lr = glob.glob(TEST_DIR + '/div2k_val_LR_AVM/2by1/qp{}/*.yuv'.format(QP))
    files_lr.sort()

    # Test GT images
    files_gt = glob.glob(TEST_DIR + '/div2k_val_HR/label/*.yuv')
    files_gt.sort()

    psnrs = []
    for ti, fn in enumerate(tqdm(files_gt)):
        # Load LR image
        W, H = _getHW(files_lr[ti], -1)
        tmp = _getYdata(files_lr[ti],[W, H],norm=False)
        img_lr = np.asarray(tmp).astype(np.float32)  # HxW

        h, w= img_lr.shape

        # Load GT image
        W, H = _getHW(files_gt[ti], -2)
        tmp = _getYdata(files_gt[ti],[W, H],norm=False)
        img_gt = np.asarray(tmp).astype(np.float32)  # HxWxC
        img_gt = img_gt[:, :, None]

        rec = superres(img_lr, 2)

        # filename = str(ti) + ".png"
        # save_img(filename,rec)

        psnr = PSNR((img_gt)[:, :, 0],rec, 0)
        print(files_gt[ti], "\n psnr:", psnr)
        psnrs.append(psnr)




    print('AVG PSNR: {}'.format(np.mean(np.asarray(psnrs))))