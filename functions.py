import skimage.io as skimage
import torch
import numpy as np
import pdb
from sewar.full_ref import sam

def matRead(data,opt):
    data=data.transpose(0,3,1,2)/2047.
    data=torch.from_numpy(data)
    data = data.to(opt.device).type(torch.cuda.FloatTensor)
    return data

def test_matRead(data,opt):
    data=data[None, :, :, :]
    data=data.transpose(0,3,1,2)/2047.
    data=torch.from_numpy(data)
    data = data.to(opt.device).type(torch.cuda.FloatTensor)
    return data

def getBatch(ms_data,pan_data,gt_data, bs):
    N = gt_data.shape[0]
    batchIndex = np.random.randint(0, N, size=bs)
    msBatch = ms_data[batchIndex, :, :, :]
    panBatch = pan_data[batchIndex, :, :, :]
    gtBatch = gt_data[batchIndex, :, :, :]
    return msBatch,panBatch,gtBatch

def getTest(ms_data,pan_data,gt_data):
    N = gt_data.shape[0]
    batchIndex = np.random.randint(0, N, size=1)
    msBatch = ms_data[batchIndex, :, :, :]
    panBatch = pan_data[batchIndex, :, :, :]
    gtBatch = gt_data[batchIndex, :, :, :]
    return msBatch,panBatch,gtBatch

def convert_image_np(inp,opt):
    inp=inp[-1,:,:,:]
    inp = inp.to(torch.device('cpu'))
    inp = inp.numpy().transpose((1,2,0))
    inp = np.clip(inp,0,1)
    inp=inp*2047.
    return inp
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def SAM(sr_img,hr_img):
    sr_img = sr_img.to(torch.device('cpu'))
    sr_img = sr_img.numpy()
    sr_img=sr_img[-1,:,:,:]
    hr_img = hr_img.to(torch.device('cpu'))
    hr_img = hr_img.numpy()
    hr_img = hr_img[-1, :, :, :]
    sam_value = sam(sr_img*1.0, hr_img*1.0)
    return sam_value

def gradientLoss_P(pan_image,opt):
    pan_image=pan_image[-1,:,:,:]
    grayGradient_x,grayGradient_y=gradient_P(pan_image)
    grayGradient_x=torch.from_numpy(grayGradient_x)
    grayGradient_x = grayGradient_x.to(opt.device).type(torch.cuda.FloatTensor)
    grayGradient_y = torch.from_numpy(grayGradient_y)
    grayGradient_y = grayGradient_y.to(opt.device).type(torch.cuda.FloatTensor)
    return grayGradient_x[None,None,:,:],grayGradient_y[None,None,:,:]

def gradient(image):
     image=image.to(torch.device('cpu'))
     image=np.array(image.detach())
     dx, dy,dz = np.gradient(image[-1,:,:,:], edge_order=1)
     image = (np.sqrt(dx ** 2 + dy ** 2+dz**2))
     return image


def gradient_P(image):
    image = image.to(torch.device('cpu'))
    image = np.array(image.detach())
    dx, dy= np.gradient(image[0, :, :], edge_order=1)
    return dx,dy