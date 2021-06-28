import json
import math
import os
# Ignore broken pipe
from signal import signal, SIGPIPE, SIG_IGN

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

import dataloaders
from dataloaders.utils.general import pix2pixHD_G, pix2pixHD_D
from nets import GAN2D
from nets.utils import MultiLayerDiscriminator, Pix2pixHDLoss, FeatLoss
from nets.utils.Blocks import Downscale
from pytorch_utils import init, util, batcher

signal(SIGPIPE, SIG_IGN)

EPS = 1e-8

parser = init.parser()
parser.add_argument('--multiGPU', action='store_true',
                    help='''Enable training on multiple GPUs, uses all that are available.''')
parser.add_argument('--load',
                    default=None,
                    help='''Load pre-trained networks''')
parser.add_argument('--net-struct', default='./structure/pix2pixHD_he-2.json',
                    help='The net structure.')

parser.add_argument('--checkpoint-dir', default="./CHECKPOINTS",
                    help='''The models are saved in this dir.''')
# Move the dataset to SSD for faster loading..
parser.add_argument('--dataset-loc',
                    default="",
                    help='Folder containing training dataset')
parser.add_argument('--name-list',
                    default="r0",
                    help='The name sequence split by _')
parser.add_argument('--label1', default='phase', help='The input image label.')
parser.add_argument('--label2', default='tdTomato', help='The ground truth image label.')
parser.add_argument('--norm-min', type=int, default=-1, help="The normalized minimum")
parser.add_argument('--norm-max', type=int, default=1, help="The normalized maximum")

parser.add_argument('--num-workers', type=int, default=3, help="The number of cores to load images.")
parser.add_argument('--crop-size', type=int, default=256, help='The input size.')
parser.add_argument('--pretrain-epoch', type=int, default=500,
                    help="The epochs that the model is pretrained with G1 output and G2 output.")
parser.add_argument('--g1', default="g1_out", help='The name of the final layer in generator 1.')
parser.add_argument('--g2', default="g2_out", help='The name of the final layer in generator 2.')
parser.add_argument('--g1-he', default="g1_heout", help='The name of the HE output layer in generator 1.')
parser.add_argument('--g2-he', default="g2_heout", help='The name of the HE output layer in generator 2.')
parser.add_argument('--d-layer', default="feat", help='The name of the final layer in discriminator.')
parser.add_argument('--n-layers', type=int, default=3, help='The levels of discriminator.')

opt = parser.parse_args()
opt.name_list = opt.name_list.split("_")
print(opt)

if not os.path.exists(opt.checkpoint_dir):
    os.makedirs(opt.checkpoint_dir)
if not os.path.exists(opt.logdir):
    os.makedirs(opt.logdir)

s = json.load(open(opt.net_struct, "rb"))
G = GAN2D.ConvNet(s["G"], [opt.g1, opt.g2, opt.g1_he, opt.g2_he]).train()
G_tmp = GAN2D.ConvNet(s["G"], opt.g2).train()
D = MultiLayerDiscriminator(s["D"], opt.d_layer, levels=opt.n_layers, para=opt.multiGPU, all_keys=True).train()
D_he = MultiLayerDiscriminator(s["D"], opt.d_layer, levels=opt.n_layers, para=opt.multiGPU, all_keys=True).train()
json.dump(s, open(os.path.join(opt.logdir, "model.json"), "w"))


train_loader = DataLoader(
    dataloaders.MyxoLabelLoaderT(opt.dataset_loc, opt.name_list, opt.label1, opt.label2, seq_len=1,
                                  mode="train", pre_process=True, crop_size=opt.crop_size),
    num_workers=opt.num_workers,  # Use this to replace data_prefetcher
    batch_size=opt.batch_size,
    shuffle=True,
    pin_memory=opt.no_cuda
)

if opt.no_cuda:
    G = G.cuda()
    G_tmp = G_tmp.cuda()
    D = D.cuda()
    D_he = D_he.cuda()

# Create model structure in tensorboard
pix2pixHD_G(G_tmp, os.path.join(opt.logdir, 'G'), opt.comment, [opt.batch_size, 1, opt.crop_size, opt.crop_size],
            use_cuda=opt.no_cuda)
del G_tmp
pix2pixHD_D(D, os.path.join(opt.logdir, 'D'), opt.comment, [opt.batch_size, 1, opt.crop_size, opt.crop_size],
            use_cuda=opt.no_cuda)

# The DataParallel should be applied after tensorboard creation
if opt.multiGPU:
    G = nn.DataParallel(G)
    D.parallel()
    D_he.parallel()

'''Load net from directory'''
if opt.load is not None:
    util.load_nets(opt.load, {
                       'G': G,
                       'D': D,
                       'D_he': D_he
                   })

# Initialize optimizer as in pix2pix paper
opt_G = optim.Adam(G.parameters(), lr=5e-5, betas=(.5, .999))
opt_D = optim.Adam(D.parameters(), lr=5e-5, betas=(.5, .999))
opt_D_he = optim.Adam(D_he.parameters(), lr=5e-5, betas=(.5, .999))

log = SummaryWriter(opt.logdir, opt.comment)
batch = batcher()

# Initialize discriminator loss and feature loss
p2pHD_loss = Pix2pixHDLoss(use_cuda=opt.no_cuda)
feat_loss = FeatLoss(num_D=opt.n_layers)

# Get the hidden features
ds = Downscale(3, 2, 1, False)  # Downscale by 2 on each edge
D.set_hook()
D_he.set_hook()


def train(epoch, alpha=None):
    if alpha is None:
        alpha = [1., 1.]
    for batch_idx in range(len(train_loader)):
        # Initialize all values
        batch.batch()
        # Prepare batch
        img0, img1, img2, target = next(train_loader.__iter__())
        if opt.no_cuda:
            img0, img1, img2 = img0.cuda(), img1.cuda(), img2.cuda()
        img0, img1, img2 = Variable(img0, requires_grad=True), Variable(img1, requires_grad=True), Variable(img2, requires_grad=True)
        g1_gen, g2_gen, g1_he, g2_he = G(img0)
        for level in range(2):
            if level == 0:
                g_idx = 0
                img_gen = g2_gen
                img_he = g2_he
            else:
                if alpha[1] == 0:
                    continue
                g_idx = 1
                img0 = ds(img0)
                img1 = ds(img1)
                img2 = ds(img2)
                img_gen = g1_gen
                img_he = g1_he
            D.start_level = g_idx
            D_he.start_level = g_idx
            feat_loss.start_level = g_idx

            # Real output and loss
            pred_fake_pool = D(torch.cat([img0, img_gen.detach()], dim=1))
            loss_D_fakes = p2pHD_loss(pred_fake_pool, False)
            pred_real = D(torch.cat([img0, img1.detach()], dim=1))
            loss_D_reals = p2pHD_loss(pred_real, True)
            pred_fake = D(torch.cat([img0, img_gen], dim=1))
            loss_G_cheat = p2pHD_loss(pred_fake, True)
            # HE output and loss
            pred_fake_pool = D_he(torch.cat([img0, img_he.detach()], dim=1))
            loss_D_fakes_he = p2pHD_loss(pred_fake_pool, False)
            pred_real_he = D_he(torch.cat([img0, img2.detach()], dim=1))
            loss_D_reals_he = p2pHD_loss(pred_real, True)
            pred_fake_he = D_he(torch.cat([img0, img_he], dim=1))
            loss_G_cheat_he = p2pHD_loss(pred_fake, True)
            loss_G_GAN_feat = feat_loss(pred_fake, pred_real) + feat_loss(pred_fake_he, pred_real_he)

            loss_D = (loss_D_fakes + loss_D_reals) * .5 * alpha[level]
            loss_D_he = (loss_D_fakes_he + loss_D_reals_he) * .5 * alpha[level]
            loss_G = (loss_G_cheat + loss_G_cheat_he + loss_G_GAN_feat * 5) * alpha[level]
            # Train D
            opt_D.zero_grad()
            loss_D.backward(retain_graph=True)
            batch.add('loss%d/%s/D_fakes' % (level, "real"), loss_D_fakes.data.item())
            batch.add('loss%d/%s/D_reals' % (level, "real"), loss_D_reals.data.item())
            batch.add('loss%d/%s/D' % (level, "real"), loss_D.data.item())
            opt_D.step()
            # Train D_he
            opt_D_he.zero_grad()
            loss_D_he.backward(retain_graph=True)
            batch.add('loss%d/%s/D_fakes' % (level, "HE"), loss_D_fakes_he.data.item())
            batch.add('loss%d/%s/D_reals' % (level, "HE"), loss_D_reals_he.data.item())
            batch.add('loss%d/%s/D' % (level, "HE"), loss_D_he.data.item())
            opt_D_he.step()
            # Train G
            opt_G.zero_grad()
            loss_G.backward(retain_graph=not (alpha[1] == 0 or level == 1))  # Only retain graph for training the whole model while training the partial model together
            batch.add('loss%d/%s/G_cheat' % (level, "real"), loss_G_cheat.data.item())
            batch.add('loss%d/%s/G_cheat' % (level, "HE"), loss_G_cheat_he.data.item())
            batch.add('loss%d/G_GAN_feat' % level, loss_G_GAN_feat.data.item())
            batch.add('loss%d/G' % level, loss_G.data.item())
            opt_G.step()

        print('\r Epoch: %d [%d/%d]: ' %
              (
                  epoch,
                  batch_idx * len(img0),
                  len(train_loader.dataset),
              ),
              end='')
        batch.report('loss/*')
        """
        TODO: Change
        """
    batch.write(log, epoch)
    print('', flush=True)
    return 0


def test(epoch, norm_range=None):
    if norm_range is None:
        norm_range = [opt.norm_min, opt.norm_max]
    MSE_list = []
    with torch.no_grad():
        img0, img1, img2, target = next(train_loader.__iter__())
        # print("\r Testing %d th batch." % batch_idx, end='')
        batch.batch()
        if opt.no_cuda:
            img0 = img0.cuda()
        _, x_fake, _, _ = G(img0)
        im1 = img1.cpu().data.numpy()
        im2 = x_fake.cpu().data.numpy()
        for i in range(np.size(im1, axis=0)):
            MSE = np.square(im1[i, 0, :, :] - im2[i, 0, :, :]).mean()
            MSE_list.append(MSE)
    MSE_mean = np.sum(MSE_list) / len(MSE_list)
    buf = make_grid(x_fake, padding=3, nrow=math.floor(math.sqrt(opt.batch_size)), normalize=True, range=(norm_range[0], norm_range[1]))
    log.add_image("Generated", buf, epoch)
    batch.write(log, epoch)
    print('Test %d: MSE %f' % (epoch, MSE_mean), flush=True)

    return MSE_mean


# Initialize training from the last record
last_epoch = 0
if opt.load:
    last_epoch = int(opt.load.split('-')[-1])


for epoch in range(last_epoch, last_epoch + opt.epochs):
    if epoch < 10 or epoch % opt.save_every == 0:
        folder_name = os.path.join(opt.checkpoint_dir, 'CHECKPOINT-%d' % epoch)
        util.save_nets(folder_name,
                       {
                           'G': G,
                           'D': D,
                           'D_he': D_he
                       })
    if epoch < opt.pretrain_epoch:
        alpha = [1., 1. - epoch / opt.pretrain_epoch]
    else:
        alpha = [1., 0.]
    train(epoch, alpha=alpha)
    if epoch < 10 or epoch % opt.save_every == 0:
        test(epoch)

epoch = last_epoch + opt.epochs
folder_name = os.path.join(opt.checkpoint_dir, 'CHECKPOINT-%d' % epoch)
util.save_nets(folder_name,
               {
                   'G': G,
                   'D': D,
                   "D_he": D_he
               })
test(epoch)
