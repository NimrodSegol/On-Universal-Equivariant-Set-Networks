import argparse
import torch.optim as optim
import torch.utils.data
from models import *
from datasets import knapsack_dataset
import utils as utils
import datetime
import os
import numpy as np
import sys
project_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_dir)
os.chdir(project_dir)


def get_percentage_within_range(model, dataset, device, A, path):
    batch = torch.from_numpy(dataset.values)
    batch = batch.to(device)
    if A is not None:
        pred = model(batch, A)
    else:
        pred = model(batch)
    pred_choice = pred.data.max(1)[1]
    pred_choice = pred_choice.view(batch.shape[0], batch.shape[1]).cpu().numpy()
    capacities = [100, 80, 50]
    epsilon = 0.1
    X = np.load(path).astype(np.float)[:, :, :4]
    X_pred = np.load(path).astype(np.float)[:, :, :4]
    y = dataset.labels
    for ii in range(4):
        X_pred[:, :, ii] *= pred_choice
    for ii in range(4):
        X[:, :, ii] *= y
    pred_capacities = X_pred[:, :, 1:].sum(axis=1)
    within_capacity = (pred_capacities <= capacities).all(1)
    true_vals = X[:, :, 0].sum(axis=1)
    pred_vals = X_pred[:, :, 0].sum(axis=1)
    values_ratio_within_epsilon = np.abs(pred_vals / true_vals - 1) <= epsilon
    successes = (values_ratio_within_epsilon * within_capacity).sum()
    return successes.item()


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--nepoch', type=int, default=150, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default=project_dir,  help='output folder')
parser.add_argument('--lr', type=float, default=0.001,  help='learning rate')
parser.add_argument('--model', type=str, default='DeepSets',  help='which architecture to use')
parser.add_argument('--optimizer', type=str, default='Adam',  help='optimization method')
parser.add_argument('--dataset_path', type=str, default=' ', help='the dataset folder')


opt = parser.parse_args()

blue = lambda x:'\033[94m' + x + '\033[0m'

torch.manual_seed(1728)
models = {'PointNet': PointNet, 'DeepSets': DeepSets, 'PointNetSeg': PointNetSeg,
          'PointNetWithOneDeepSetLayer': PointNetWithOneDeepSetLayer,  'MLP': MLP,
          'PointNetWithPolynomialLayer': PointNetWithPolynomialLayer, "GraphNet": GraphNet}
gr = False
if opt.model == "GraphNet":
    gr = True
dataset = knapsack_dataset(opt.dataset_path, train=True, graph=gr)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))

test_dataset = knapsack_dataset(opt.dataset_path, train=False, graph=gr)

test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,
                                               shuffle=False, num_workers=int(opt.workers))

num_classes = int(dataset.labels.max() + 1)
num_points = dataset[0][0].shape[0]
num_batch = len(dataset) / opt.batchSize

if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)


for width in np.linspace(100, 5, 15).astype(np.int):
    classifier = models[opt.model](depth=6, width=width, in_features=4,  out_features=2)

    optimizer = optim.Adam(classifier.parameters(), lr=opt.lr, betas=(0.9, 0.999))
    if opt.optimizer == 'SGD':
        optimizer = optim.SGD(classifier.parameters(), lr=opt.lr, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    classifier = classifier.to(device)

    now = datetime.datetime.now()
    folder_name = os.path.join('knapsack_log', opt.model, 'width'+str(width))

    start_run_time = str(now.day) + str(now.month) + str(now.hour) + str(now.minute)
    filename = 'optimize=' + str(optimizer)[:3] + 'lr=' + str(opt.lr) + '.csv'
    utils.create_dir_if_not_exist(os.path.join(opt.outf, folder_name))

    logpath = os.path.join(opt.outf, folder_name, filename)
    cols = ['epoch', 'train_loss', 'train_success_percentage', 'test_loss', 'test_success_percentage']
    logger = utils.Logger(logpath, cols)

    for epoch in range(opt.nepoch):
        scheduler.step()
        train_loss, train_acc, test_loss, test_acc = 0.0, 0.0, 0.0, 0.0
        for i, data in enumerate(data_loader, 0):
            if opt.model == "GraphNet":
                points, A, target = data
                A = A.float()
                A = A.to(device)
            else:
                points, target = data
                A = None
            points, target = points.to(device), target.to(device)
            target = target.squeeze(1)
            optimizer.zero_grad()
            classifier = classifier.train()
            if opt.model == "GraphNet":
                pred = classifier(points, A)
            else:
                pred = classifier(points)
            target = target.view(-1, 1)[:, 0]
            loss = F.nll_loss(pred, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if i % 50 == 0:
                print('[%s %s - %d: %d/%d] train loss: %f' % (opt.model, filename, epoch, i, num_batch, loss.item()))

        with torch.no_grad():
            for j, data in (enumerate(test_data_loader, 0)):
                if opt.model == "GraphNet":
                    points, A, target = data
                    A = A.float()
                    A = A.to(device)
                else:
                    points, target = data
                points, target = points.to(device), target.to(device)
                target = target.squeeze(1)
                classifier = classifier.eval()
                if opt.model == "GraphNet":
                    pred = classifier(points, A)
                else:
                    pred = classifier(points)
                    A = None
                pred = pred.view(-1, num_classes)
                target = target.view(-1, 1)[:, 0]
                loss = F.nll_loss(pred, target)
                test_loss += loss.item()
                print('[%d: %d/%d] %s loss: %f' % (epoch, i, len(test_dataset)/opt.batchSize, blue('test'),
                                                   loss.item()))
        path = os.path.join(opt.dataset_path, 'dataset_4feat.npy')
        train_acc = get_percentage_within_range(classifier, dataset, device, A, path)
        print(train_acc)
        path = os.path.join(opt.dataset_path, 'test_dataset_4feat.npy')
        test_acc = get_percentage_within_range(classifier, test_dataset, device, A, path)
        logger.log(
            {
                'epoch': epoch,
                'train_loss': train_loss/float(len(dataset)),
                'train_success_percentage': train_acc/float(len(dataset)),
                'test_loss': test_loss/float(len(test_dataset)),
                'test_success_percentage': test_acc/float(len(test_dataset)),
            }
        )

    folder = os.path.join(opt.outf, folder_name, 'knapsack_model')
    utils.create_dir_if_not_exist(folder)
    torch.save(classifier.state_dict(), '%s/%s.pth' % (folder, filename))
