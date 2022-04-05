from os.path import join
import os
import torch
from torch.utils.data import Dataset
from skimage import io as skimage

def matRead(data):
    data=data.transpose(2,0,1)/2047.
    data=torch.from_numpy(data)
    return data

class Dataset(Dataset):

    def __init__(self, path):
        super(Dataset, self).__init__()
        self.panpath = path + 'SimulatedRawPAN/'
        self.mspath = path + 'SimulatedRawMS/'
        self.refpath = path + 'ReferenceRawMS/'

        files = os.listdir(self.panpath)
        img_list = []
        for index in files:
            num = index.split('p')[0]
            img_list.append(num)
        self.img_list = img_list

    def __getitem__(self, index):
        fn = self.img_list[index]

        panBatch = skimage.imread(self.panpath + str(fn) + 'p.tif')
        panBatch = panBatch[:, :, None]
        panBatch = matRead(panBatch)

        msBatch = skimage.imread(self.mspath + str(fn) + 'ms.tif')
        msBatch = matRead(msBatch)

        gtBatch = skimage.imread(self.refpath + str(fn) + 'MSref.tif')
        gtBatch = matRead(gtBatch)

        return gtBatch, msBatch, panBatch

    def __len__(self):
        return len(self.img_list)

def get_training_set(train_dir):
    return Dataset(train_dir)

def get_val_set(val_dir):
    return Dataset(val_dir)