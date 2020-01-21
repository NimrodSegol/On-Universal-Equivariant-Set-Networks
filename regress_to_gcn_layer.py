import os
import numpy as np
import datetime
import torch.optim as optim
from models import *
import torch
import torch.nn as nn
import utils as utils
import torch.utils.data as Data
import argparse
import sys


project_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_dir)
os.chdir(project_dir)


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001,  help='learning rate')
parser.add_argument('--num_exp', type=int, default=1000,  help='examples')
parser.add_argument('--num_points', type=int, default=100,  help='examples')
parser.add_argument('--outf', type=str, default=project_dir,  help='output folder')

parser.add_argument('--n_feat', type=int, default=3,  help='features')

opt = parser.parse_args()

#  create dataset


n_examples, n_points, n_features = opt.num_exp, opt.num_points, opt.n_feat

torch.manual_seed(1728)

x = torch.randn(n_examples, n_points, n_features) + 0.5
A = utils.make_batch_of_adj_matrices(x, k=10)
gcn = GraphConvolution(n_features, 10)
with torch.no_grad():
    y = gcn(x, A)
for width in np.linspace(13, 200, 8).astype(np.int):
    net = DeepSets(depth=6, width=width, in_features=n_features, out_features=10, regression=True)
    net = net.cuda()
    optimizer = optim.Adam(net.parameters(), lr=opt.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    now = datetime.datetime.now()
    folder_name = 'gcn_regress_width' + str(width)

    start_run_time = str(now.day) + str(now.month) + str(now.hour) + str(now.minute)
    filename = 'optimize=' + str(optimizer)[:3] + 'lr=' + str(opt.lr) + 'start_run_time=' + start_run_time + '.csv'

    utils.create_dir_if_not_exist(os.path.join(opt.outf, folder_name))

    logpath = os.path.join(opt.outf, folder_name, filename)
    cols = ['epoch', 'loss']
    logger = utils.Logger(logpath, cols)
    batch_size = 32
    torch_dataset = Data.TensorDataset(x, y)

    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=batch_size,
        shuffle=True, num_workers=1, )

    loss_func = nn.SmoothL1Loss()
    nepoch = 200
    for epoch in range(nepoch):
        epoch_loss = 0.0
        # scheduler.step()
        for step, (batch_x, batch_y) in enumerate(loader):  # for each training step
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()  # move variables to gpu
            net = net.train()  # network to train mode
            optimizer.zero_grad()  # clear gradients for next train
            prediction = net(batch_x) # input x and predict based on x
            loss = loss_func(prediction, batch_y)  # must be (1. nn output, 2. target)
            epoch_loss += loss.item() * (batch_x.shape[0])
            loss.backward()  # back-propagation, compute gradients
            optimizer.step()  # apply gradients
        epoch_loss = epoch_loss/float(n_examples)
        print("epoch %d , loss %f" % (epoch, epoch_loss))
        logger.log(
            {
                'epoch': epoch,
                'loss': epoch_loss,
            }
        )




