from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from imageio.v2 import imread
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
# from torchmetrics import StructuralSimilarityIndexMeasure as ssim
# from torchmetrics import PeakSignalNoiseRatio as psnr

import argparse
import glob
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument("--train_dir", help="path to folder containing training set")
parser.add_argument("--test_dir", help="path to folder containing testing set")
parser.add_argument("--max_epochs", type=int, default=1, help="number of training epochs")
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
parser.add_argument("--direction", type=str, default="lr", choices=["lr", "rl"], help="which direction")

args = parser.parse_args()

class Dataset(data.Dataset):
    def __init__(self, image_dir, labels_dir):
        super(Dataset, self).__init__()
        # self.path = image_dir
        self.input_filenames = glob.glob(os.path.join(image_dir, "*.png"))
        self.label_filenames = glob.glob(os.path.join(labels_dir, "*.png"))

    def __getitem__(self, index):
        # Load Image
        img = np.transpose(imread(self.input_filenames[index]), (2,0,1)).astype(np.float32) / 255.0
        label = np.transpose(imread(self.label_filenames[index]), (2,0,1)).astype(np.float32) / 255.0
        left = torch.from_numpy(img)
        right = torch.from_numpy(label)

        return left, right

    def __len__(self):
        return len(self.input_filenames)

# model init
from models import weights_init, _netG, _netD

netG = _netG(input_nc=3, target_nc=3, ngf=64)
netG.apply(weights_init)

netD = _netD(input_nc=3, target_nc=3, ndf=64)
netD.apply(weights_init)

input = torch.FloatTensor(args.batch_size, 3, 1080, 1920)
target = torch.FloatTensor(args.batch_size, 3, 1080, 1920)



# move to gpu
if args.cuda:
    netD = netD.cuda()
    netG = netG.cuda()
    input = input.cuda()
    target = target.cuda()
    # ones_label = ones_label.cuda()
    # zeros_label = zeros_label.cuda()

# convert to Variable
input = Variable(input)
target = Variable(target)
# ones_label = Variable(ones_label)
# zeros_label = Variable(zeros_label)

# optimizer
D_solver = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
G_solver = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

# load data
train_images_dir = os.path.join(args.train_dir,"banded")
train_labels_dir = os.path.join(args.train_dir,"pristine")
test_images_dir = os.path.join(args.test_dir,"banded")
test_labels_dir = os.path.join(args.test_dir,"pristine")
train_set = Dataset(train_images_dir,train_labels_dir)
training_data_loader = data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
test_set = Dataset(test_images_dir, test_labels_dir)
testing_data_loader = data.DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False)

def train(epoch):
    avg_psnr = 0
    avg_ssim = 0
    for batch, (left, right) in enumerate(training_data_loader):
        if args.direction == 'lr':
            input.data.resize_(left.size()).copy_(left)
            target.data.resize_(right.size()).copy_(right)
        else:
            input.data.resize_(right.size()).copy_(right)
            target.data.resize_(left.size()).copy_(left)

        ## Discriminator
        netD.zero_grad()
        # real
        D_real = netD(input, target)
        ones_label = torch.ones_like(D_real)
        zeros_label = torch.zeros_like(D_real)
        D_loss_real = F.binary_cross_entropy(D_real, ones_label)
        D_x_y = D_real.data.mean()

        # fake
        G_fake = netG(input)
        D_fake = netD(input, G_fake.detach())
        D_loss_fake = F.binary_cross_entropy(D_fake, zeros_label)
        D_x_gx = D_fake.data.mean()

        D_loss = D_loss_real + D_loss_fake
        D_loss.backward()
        D_solver.step()

        ## Generator
        netG.zero_grad()

        G_fake = netG(input)
        D_fake = netD(input, G_fake)
        D_x_gx_2 = D_fake.data.mean()
        G_loss = F.binary_cross_entropy(D_fake, ones_label) + 100 * F.smooth_l1_loss(G_fake, target)
        G_loss.backward()
        G_solver.step()
        im_true = np.transpose(target.data.cpu().numpy(), (0, 2, 3, 1))
        im_test = np.transpose(G_fake.data.cpu().numpy(), (0, 2, 3, 1))
        for i in range(input.size(0)):
            avg_psnr += psnr(im_true[i], im_test[i])
            avg_ssim += (ssim(im_true[i,:,:,0], im_test[i,:,:,0], data_range=1.0) +
             ssim(im_true[i,:,:,1], im_test[i,:,:,1], data_range=1.0) +
             ssim(im_true[i,:,:,2], im_test[i,:,:,2], data_range=1.0)) / 3

        ## debug
        if (batch + 1) % 100 == 0:
            print('[TRAIN] Epoch[{}]({}/{}); D_loss: {:.4f}; G_loss: {:.4f}; D(x): {:.4f} D(G(z)): {:.4f}/{:.4f}'.format(
                epoch, batch + 1, len(training_data_loader), D_loss.data[0], G_loss.data[0], D_x_y, D_x_gx, D_x_gx_2))

def test(epoch):
    avg_psnr = 0
    avg_ssim = 0
    for left, right in testing_data_loader:

        if args.direction == 'lr':
            input.data.resize_(left.size()).copy_(left)
            target.data.resize_(right.size()).copy_(right)
        else:
            input.data.resize_(right.size()).copy_(right)
            target.data.resize_(left.size()).copy_(left)

        prediction = netG(input)

        im_true = np.transpose(target.data.cpu().numpy(), (0, 2, 3, 1))
        im_test = np.transpose(prediction.data.cpu().numpy(), (0, 2, 3, 1))
        for i in range(input.size(0)):
            avg_psnr += psnr(im_true[i], im_test[i])
            avg_ssim += (ssim(im_true[i,:,:,0], im_test[i,:,:,0], data_range=1.0) +
             ssim(im_true[i,:,:,1], im_test[i,:,:,1], data_range=1.0) +
             ssim(im_true[i,:,:,2], im_test[i,:,:,2], data_range=1.0)) / 3

    print("[TEST]  PSNR: {:.4f}; SSIM: {:.4f}".format(avg_psnr / len(test_set), avg_ssim / len(test_set)))


def main():
    for epoch in range(1, args.max_epochs + 1):
        train(epoch)
        #test(epoch)
        if not os.path.exists("checkpoint"):
            os.mkdir("checkpoint")
        if epoch % 10 == 0:
            torch.save(netG, 'checkpoint/netG_epoch_{}_{:03d}.pth'.format(args.direction, epoch))
            torch.save(netD, 'checkpoint/netD_epoch_{}_{:03d}.pth'.format(args.direction, epoch))
main()
