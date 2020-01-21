import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class DeepSetLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(DeepSetLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.layer1 = nn.Conv1d(in_features, out_features, 1)
        self.layer2 = nn.Conv1d(in_features, out_features, 1, bias=False)

    def forward(self, x):
        return self.layer1(x) + self.layer2(x - x.mean(dim=2, keepdim=True))


class PolyEquiLayer(nn.Module):
    def __init__(self, in_feat):
        super(PolyEquiLayer, self).__init__()
        self.conv1 = nn.Conv1d(in_feat, in_feat, 1)
        self.conv2 = nn.Conv1d(in_feat, in_feat, 1, bias=False)
        self.conv3 = nn.Conv1d(in_feat, in_feat, 1, bias=False)
        self.conv4 = nn.Conv1d(in_feat, in_feat, 1, bias=False)
        self.conv5 = nn.Conv1d(in_feat, in_feat, 1, bias=False)

    def forward(self, x):
        #dims batch, num_features, num_points
        n_pts = x.shape[2]
        return self.conv1(x) + self.conv2(x - x.mean(dim=2, keepdim=True))\
               + self.conv3((x.mean(dim=2,  keepdim=True) ** 2).repeat(1, 1, n_pts))\
               + self.conv4((x**2).sum(dim=2, keepdim=True).repeat(1, 1, n_pts))/n_pts\
               + self.conv5(x.mean(dim=2, keepdim=True)*x)


class PointNet(nn.Module):
    def __init__(self, depth, width,  in_features,  out_features=2, regression=False):
        super(PointNet, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.regression = regression

        self.first_layer = nn.Conv1d(self.in_features, width, 1)
        self.hidden = nn.ModuleList()
        for _ in range(depth - 2):
            self.hidden.append(nn.Conv1d(width, width, 1))
        self.last_layer = nn.Conv1d(width, self.out_features, 1)

    def forward(self, x):
        x = x.transpose(2, 1)
        # the dims should now be BxFxn
        x = F.relu(self.first_layer(x))

        for layer in self.hidden:
            x = F.relu(layer(x))

        x = self.last_layer(x)
        x = x.transpose(2, 1).contiguous()
        if self.regression:
            return x
        else:
            x = F.log_softmax(x.view(-1, self.out_features), dim=-1)
            return x


class DeepSets(nn.Module):
    def __init__(self, depth, width,  in_features,  out_features=2, regression=False):
        super(DeepSets, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.regression = regression

        self.first_layer = DeepSetLayer(self.in_features, width)
        self.hidden = nn.ModuleList()
        for _ in range(depth - 2):
            self.hidden.append(DeepSetLayer(width, width))
        self.last_layer = DeepSetLayer(width, self.out_features)

    def forward(self, x):
        x = x.transpose(2, 1)
        # the dims should now be BxFxn
        x = F.relu(self.first_layer(x))

        for layer in self.hidden:
            x = F.relu(layer(x))

        x = self.last_layer(x)
        x = x.transpose(2, 1).contiguous()
        if self.regression:
            return x
        else:
            x = F.log_softmax(x.view(-1, self.out_features), dim=-1)
            return x


class PointNetWithPolynomialLayer(nn.Module):
    def __init__(self, depth, width,  in_features,  out_features=2, regression=False):
        super(PointNetWithPolynomialLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.regression = regression

        self.first_layer = nn.Conv1d(self.in_features, width, 1)
        self.first_hidden = nn.ModuleList()
        for _ in range(int(depth/2) - 1):
            self.first_hidden.append(nn.Conv1d(width, width, 1))
        self.poly = PolyEquiLayer(width)
        self.second_hidden = nn.ModuleList()
        for _ in range(int(depth / 2) - 2):
            self.second_hidden.append(nn.Conv1d(width, width, 1))
        self.last_layer = nn.Conv1d(width, self.out_features, 1)

    def forward(self, x):
        x = x.transpose(2, 1)
        # the dims should now be BxFxn
        x = F.relu(self.first_layer(x))
        for layer in self.first_hidden:
            x = F.relu(layer(x))
        x = F.relu(self.poly(x))
        for layer in self.second_hidden:
            x = F.relu(layer(x))
        x = self.last_layer(x)
        x = x.transpose(2, 1).contiguous()
        if self.regression:
            return x
        else:
            x = F.log_softmax(x.view(-1, self.out_features), dim=-1)
            return x


class PointNetSeg(nn.Module):
    def __init__(self,  depth, width,  in_features,  out_features=2, regression=False):
        super(PointNetSeg, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.regression = regression

        self.first_layer = nn.Conv1d(self.in_features, width, 1)
        self.first_hidden = nn.ModuleList()
        for _ in range(int(depth / 2) - 1):
            self.first_hidden.append(nn.Conv1d(width, width, 1))
        self.middle_layer = nn.Conv1d(2*width, width, 1)
        self.second_hidden = nn.ModuleList()
        for _ in range(int(depth / 2) - 2):
            self.second_hidden.append(nn.Conv1d(width, width, 1))
        self.last_layer = nn.Conv1d(width, self.out_features, 1)

    def forward(self, x):
        x = x.transpose(2, 1)
        # the dims should now be BxFxn
        n_pts = x.shape[2]
        x = F.relu(self.first_layer(x))
        temp_x_for_concat = x
        for layer in self.first_hidden:
            x = F.relu(layer(x))
        x = torch.cat((temp_x_for_concat, x.mean(dim=2, keepdim=True).repeat(1, 1, n_pts)), dim=1)
        x = F.relu(self.middle_layer(x))
        for layer in self.second_hidden:
            x = F.relu(layer(x))
        x = self.last_layer(x)
        x = x.transpose(2, 1).contiguous()
        if self.regression:
            return x
        else:
            x = F.log_softmax(x.view(-1, self.out_features), dim=-1)
            return x


class PointNetWithOneDeepSetLayer(nn.Module):
    def __init__(self, depth, width,  in_features,  out_features=2, regression=False):
        super(PointNetWithOneDeepSetLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.regression = regression

        self.first_layer = nn.Conv1d(self.in_features, width, 1)
        self.first_hidden = nn.ModuleList()
        for _ in range(int(depth / 2) - 1):
            self.first_hidden.append(nn.Conv1d(width, width, 1))
        self.middle_layer = DeepSetLayer(width, width)
        self.second_hidden = nn.ModuleList()
        for _ in range(int(depth / 2) - 2):
            self.second_hidden.append(nn.Conv1d(width, width, 1))
        self.last_layer = nn.Conv1d(width, self.out_features, 1)

    def forward(self, x):
        x = x.transpose(2, 1)
        # the dims should now be BxFxn
        n_pts = x.shape[2]
        x = F.relu(self.first_layer(x))
        for layer in self.first_hidden:
            x = F.relu(layer(x))
        x = F.relu(self.middle_layer(x))
        for layer in self.second_hidden:
            x = F.relu(layer(x))
        x = self.last_layer(x)
        x = x.transpose(2, 1).contiguous()
        if self.regression:
            return x
        else:
            x = F.log_softmax(x.view(-1, self.out_features), dim=-1)
            return x


class MLP(nn.Module):
    def __init__(self, depth, width,  in_features,  out_features=2, regression=False):
        super(MLP, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.regression = regression

        self.first_layer = nn.Linear(self.in_features*512, width)
        self.hidden = nn.ModuleList()
        for _ in range(depth - 2):
            self.hidden.append(nn.Linear(width, width))
        self.last_layer = nn.Linear(width, self.out_features*512)

    def forward(self, x):
        # x = x.transpose(2, 1)
        # the dims should now be BxnxF !!unlike other models!!
        n_pts = x.shape[1]
        x = x.view(x.shape[0], n_pts*self.in_features)
        x = F.relu(self.first_layer(x))

        for layer in self.hidden:
            x = F.relu(layer(x))

        x = self.last_layer(x)
        x = x.view(x.shape[0], n_pts, self.out_features)
        if self.regression:
            return x
        else:
            x = F.log_softmax(x.view(-1, self.out_features), dim=-1)
            return x


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lin = nn.Linear(in_features, out_features)
        self.weight = Parameter(torch.randn(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.bmm(adj, support)+self.lin(input)
        return output


class GraphNet(nn.Module):
    def __init__(self, depth, width, in_features,  out_features=2, regression=False):
        super(GraphNet, self).__init__()

        self.out_features = out_features
        self.regression = regression
        self.convs = nn.ModuleList()
        self.convs.append(GraphConvolution(in_features, width))
        for i in range(1, depth - 1):
            self.convs.append(GraphConvolution(width, width))
        self.convs.append(GraphConvolution(width, out_features))

    def forward(self, x, A):
        # the dims should now be BxnxF
        for i in range(len(self.convs) - 1):
            x  = F.relu(self.convs[i](x, A))
        x = F.relu(self.convs[-1](x, A))
        if self.regression:
            return x
        else:
            x = F.log_softmax(x.view(-1, self.out_features), dim=-1)
            return x

