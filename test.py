from torch.autograd import Variable
import model
import torch
import functions
import numpy
import os
from skimage import io
import argparse
import scipy.io

def test_matRead(data):
    data=data[None, :, :, :]
    data=data.transpose(0,3,1,2)/2047.
    data=torch.from_numpy(data)
    data = data.to(torch.device('cuda:0')).type(torch.cuda.FloatTensor)
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mspath', help='test lrms image name')# default='')
    parser.add_argument('--panpath', help='test hrpan image name')# default='')
    parser.add_argument('--modelpath', help='output model dir')# default='')
    parser.add_argument('--saveimgpath', help='output model dir')# default='')
    parser.add_argument('--device', default=torch.device('cuda:0'))
    parser.add_argument('--nfc', type=int, default=32)
    parser.add_argument('--nfc64', type=int, default=64)
    opt = parser.parse_args()

    net = model.Net(opt).to(opt.device)

    modelname = opt.modelpath
    net.load_state_dict(torch.load(modelname))
    for msfilename in os.listdir(opt.mspath):
        num = msfilename.split('m')[0]
        print(opt.mspath + msfilename)
        ms_val = io.imread(opt.mspath + msfilename)
        ms_val = test_matRead(ms_val)
        ms_val = torch.nn.functional.interpolate(ms_val, size=(256, 256), mode='bilinear')
        panname = msfilename.split('m')[0]+'p.tif' #'p.tif'
        pan_val = io.imread(opt.panpath + panname)
        pan_val = pan_val[:, :, None]
        pan_val = test_matRead(pan_val)
        _,_,in_s = net(ms_val, pan_val)
        outname = opt.saveimgpath + num +'.tif'
        io.imsave(outname, functions.convert_image_np(in_s.detach(), opt).astype(numpy.uint16))

if __name__ == '__main__':
    main()