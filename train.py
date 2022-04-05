import argparse
import model as model
import torch
import torch.nn as nn
import functions
import time
import os
import copy
import random
import dataloader
from torch.utils.data import DataLoader
from torch.autograd import Variable

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help='input image dir')#, default='')
    parser.add_argument('--test_dir', help='testing_data')#, default='')
    parser.add_argument('--outputs_dir',help='output model dir')#, default='')
    parser.add_argument('--batchSize', default=16)
    parser.add_argument('--testBatchSize', default=1)
    parser.add_argument('--epoch', default=300)  # 1300)
    parser.add_argument('--not_cuda', action='store_true', help='disables cuda', default=0)
    parser.add_argument('--device',default=torch.device('cuda:1'))
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--lr',type=float,default=0.0001,help='G‘s learning rate')
    parser.add_argument('--gamma',type=float,default=0.01,help='scheduler gamma')
    parser.add_argument('--nfc',type=int,default=32)
    parser.add_argument('--nfc64',type=int,default=64)
    parser.add_argument('--weight_gradient',type=float,default=0.001)
    parser.add_argument('--weight_L',type=float,default=0.001)
    opt = parser.parse_args()

    seed = random.randint(1, 10000)
    torch.manual_seed(seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = False

    train_set = dataloader.get_training_set(opt.input_dir)
    val_set = dataloader.get_val_set(opt.test_dir)

    train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize,
                              shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

    # 网络初始化：
    net = model.Net(opt).to(opt.device)
    for module in net.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

    # 建立优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,milestones=[1600],gamma=opt.gamma)

    loss = torch.nn.L1Loss()
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        net = net.to(opt.device)
        loss = loss.to(opt.device)
    best_weights = copy.deepcopy(net.state_dict())
    best_epoch = 0
    best_SAM=1.0
    for i in range(opt.epoch):
        # train
        net.train()
        epoch_losses = functions.AverageMeter()
        batch_time = functions.AverageMeter()
        end = time.time()
        for batch_idx, (gtBatch, msBatch, panBatch) in enumerate(train_loader):

            if torch.cuda.is_available():
                msBatch, panBatch, gtBatch = msBatch.to(opt.device), panBatch.to(opt.device), gtBatch.to(opt.device)
                msBatch = Variable(msBatch.to(torch.float32))
                panBatch = Variable(panBatch.to(torch.float32))
                gtBatch = Variable(gtBatch.to(torch.float32))
            msBatch,panBatch,gtBatch=functions.getBatch(msBatch,panBatch,gtBatch,opt.batchSize)
            N = len(train_loader)
            net.zero_grad()
            msBatch = torch.nn.functional.interpolate(msBatch, size=(gtBatch.shape[2], gtBatch.shape[3]),
                                                      mode='bilinear')
            ms_, pan_, out = net(msBatch, panBatch)
            outLoss = loss(out, gtBatch)
            outLoss.backward(retain_graph=True)
            msLoss = loss(ms_, msBatch)
            panLoss = loss(pan_, panBatch)
            ######## gradient loss x+y ##########
            pan_Batch_gradient_x, pan_Batch_gradient_y = functions.gradientLoss_P(panBatch,opt)
            pan_image_gradient_x, pan_image_gradient_y = functions.gradientLoss_P(pan_,opt)
            gradient_loss_x = opt.weight_gradient * loss(pan_Batch_gradient_x.requires_grad_(),
                                                         pan_image_gradient_x.requires_grad_())
            gradient_loss_y = opt.weight_gradient * loss(pan_Batch_gradient_y.requires_grad_(),
                                                         pan_image_gradient_y.requires_grad_())
            (panLoss + gradient_loss_y + gradient_loss_x).backward(retain_graph=True)
            # ms波段相关性loss
            numloss=0
            rel_tol = 1e-09
            for m in range(4 - 1):
                for n in range(m + 1, 4):
                    outChannel = ms_[:, n, :, :]/(ms_[:, m, :, :]+rel_tol)
                    tureChannel = msBatch[:, n, :, :]/(msBatch[:, m, :, :]+rel_tol)
                    channel_loss=loss(outChannel,tureChannel)
                    numloss=numloss+channel_loss
                numloss+=numloss
            (numloss + msLoss).backward(retain_graph=True)
            optimizer.step()
            epoch_losses.update(msLoss.item(), msBatch.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if (batch_idx + 1) % 100 == 0:
                training_state = '  '.join(
                    ['Epoch: {}', '[{} / {}]', 'msLoss: {:.6f}','panLoss: {:.6f}']
                )
                training_state = training_state.format(
                    i, batch_idx, N, msLoss, panLoss
                )
                print(training_state)

        print('%d epoch: loss is %.6f, epoch time is %.4f' % (i, epoch_losses.avg, batch_time.avg))
        torch.save(net.state_dict(), os.path.join(opt.outputs_dir, 'epoch_{}.pth'.format(i)))
        net.eval()
        epoch_SAM=functions.AverageMeter()
        with torch.no_grad():
            for j, (gtTest, msTest, panTest) in enumerate(val_loader):
                if torch.cuda.is_available():
                    msTest, panTest, gtTest = msTest.to(opt.device), panTest.to(opt.device), gtTest.to(opt.device)
                    msTest = Variable(msTest.to(torch.float32))
                    panTest = Variable(panTest.to(torch.float32))
                    gtTest = Variable(gtTest.to(torch.float32))
                    net = net.to(opt.device)
                msTest = torch.nn.functional.interpolate(msTest, size=(256, 256), mode='bilinear')
                _,_,mp = net(msTest, panTest)
                test_SAM=functions.SAM(mp, gtTest)
                if test_SAM==test_SAM:
                    epoch_SAM.update(test_SAM,msTest.shape[0])
            print('eval SAM: {:.6f}'.format(epoch_SAM.avg))

        if epoch_SAM.avg < best_SAM:
            best_epoch = i
            best_SAM = epoch_SAM.avg
            best_weights = copy.deepcopy(net.state_dict())
        print('best epoch:{:.0f}'.format(best_epoch))
        scheduler.step()

    print('best epoch: {}, epoch_SAM: {:.6f}'.format(best_epoch, best_SAM))
    torch.save(best_weights, os.path.join(opt.outputs_dir, 'best.pth'))