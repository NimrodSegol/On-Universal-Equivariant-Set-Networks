import argparse
import torch.optim as optim
from models import *
import torch
from datasets import Equidataset
import utils as utils
import datetime
import os
import numpy as np
import sys
project_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_dir)
os.chdir(project_dir)

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--outf', type=str, default=project_dir,  help='output folder')
parser.add_argument('--path', type=str, default=project_dir,  help='where are the dataset')

parser.add_argument('--lr', type=float, default=0.001,  help='learning rate')
parser.add_argument('--model', type=str, default='PointNet',  help='model')
parser.add_argument('--optimizer', type=str, default='Adam',  help='optimization method')


opt = parser.parse_args()


torch.manual_seed(1728)

gr = False
if opt.model == 'GraphNet':
    gr = True

n = 100
x = torch.randn(10000, 100, 4) - 0.5
y = ((x - 0.5)**2).sum(dim=2).sum(dim=1, keepdim=True).repeat(1, 100)
dataset = Equidataset(x, y, graph=gr)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))
test_x = torch.randn(1000, 100, 4) - 0.5
test_y = ((x - 0.5)**2).sum(dim=2).sum(dim=1, keepdim=True).repeat(1, 100)
test_dataset = Equidataset(test_x, test_y, graph=gr)

testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,
                                             shuffle=False, num_workers=int(opt.workers))


models = {'PointNet': PointNet, 'DeepSets': DeepSets, 'PointNetSeg': PointNetSeg,
          'PointNetWithOneDeepSetLayer': PointNetWithOneDeepSetLayer, 'MLP': MLP,
          'PointNetWithPolynomialLayer': PointNetWithPolynomialLayer, "GraphNet": GraphNet}
if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)
num_classes = 1
n_features = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for width in np.linspace(256*3, 5,  15).astype(np.int):
    net = models[opt.model](depth=6, width=width, in_features=n_features, out_features=1, regression=True)
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=opt.lr, betas=(0.9, 0.999))
    if opt.optimizer == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    now = datetime.datetime.now()
    folder_name = os.path.join('regression_log', opt.model, 'width' + str(width))

    start_run_time = str(now.day) + str(now.month) + str(now.hour) + str(now.minute)
    filename = 'optimize=' + str(optimizer)[:3] + '.csv'

    utils.create_dir_if_not_exist(os.path.join(opt.outf, folder_name))

    logpath = os.path.join(opt.outf, folder_name, filename)
    cols = ['epoch', 'train_loss', 'test_loss']
    logger = utils.Logger(logpath, cols)

    loss_func = nn.MSELoss()
    nepoch = 50

    for epoch in range(nepoch):
        epoch_loss = 0.0
        # scheduler.step()
        for step, data in enumerate(dataloader):  # for each training step
            if opt.model == "GraphNet":
                batch_x, A, batch_y = data
                A = A.float()
                A = A.to(device)
            else:
                batch_x, batch_y = data
                A = None
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            net = net.train()  # network to train mode
            optimizer.zero_grad()  # clear gradients for next train
            if opt.model == "GraphNet":
                prediction = net(batch_x, A).squeeze(2)
            else:
                prediction = net(batch_x).squeeze(2)
            loss = loss_func(prediction, batch_y)  # must be (1. nn output, 2. target)
            epoch_loss += loss.item() * (batch_x.shape[0])
            loss.backward()  # back-propagation, compute gradients
            optimizer.step()  # apply gradients
            if step % 10 == 0:
                print("epoch %d %s loss %f" % (epoch, logger.logfilepath, loss.item() * (batch_x.shape[0])))

        with torch.no_grad():
            test_loss = 0.0
            net = net.eval()
            for _, data in enumerate(testdataloader):
                if opt.model == "GraphNet":
                    test_batch_x, A, test_batch_y = data
                    A = A.float()
                    A = A.to(device)
                else:
                    test_batch_x, test_batch_y = data
                    A = None
                test_batch_x, test_batch_y = test_batch_x.to(device), test_batch_y.to(device)
                if opt.model == "GraphNet":
                    pred = net(test_batch_x, A).squeeze(2)
                else:
                    pred = net(test_batch_x).squeeze(2)
                loss = loss_func(pred, test_batch_y)
                test_loss += loss.item()*(test_batch_x.shape[0])
            epoch_loss = epoch_loss / float(len(dataset))
            test_loss = test_loss / float(len(test_dataset))
            print(" epoch %d  test_loss %f" % (epoch, test_loss))
            logger.log(
                {
                    'epoch': epoch,
                    'train_loss': epoch_loss,
                    'test_loss': test_loss,
                }
            )

