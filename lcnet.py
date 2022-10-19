import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from moic.mklearn.nn.functional_net import FCBlock

import networkx as nx

from config import *

eps = 5

def L2Norm(x,y):
    if x is not torch.tensor: x = torch.tensor(x)
    if y is not torch.tensor: y = torch.tensor(y)
    return torch.sqrt(x*x + y*y)

def same_circle(a,b):
    x1,y1,r1 = a; x2,y2,r2 = b
    if L2Norm(x1-x2,y1-y2) > eps or torch.abs(r1-r2)> eps:return False
    return True

def same_line(a,b):
    start1,end1 = a;start2,end2 = b
    xs1,ys1 = start1;xe1,ye1 = end1;
    xs2,ys2 = start2;xe2,ye2 = end2;
    if L2Norm(xs1-xs2,ys1-ys2) > eps or L2Norm(xe1-xe2,ye1-ye2) > eps:return False
    return True

class LCNet(nn.Module):
    def __init__(self,opt = model_opt):
        super().__init__()
        self.line_count   = 0
        self.circle_count = 0
        self.point_count  = 0

        # record features from the input lines and circles
        self.line_embeddings   = {}
        self.circle_embeddings = {}
        # after the propagation

        # construct the crude rpresentation of input lines and circles
        self.lines   = {}
        self.circles = {}

        # the name of the points will be generated automatically
        self.graph = nx.DiGraph()

        self.net_data = 0
        
        # the embeeding 
        self.conv1  =  GCNConv(64,  128)
        self.conv2  =  GCNConv(128, 128)
        self.conv3  =  GCNConv(128, 64)
        # use the graph convolution to map the data from the point and circle
        self.line_mapper    = FCBlock(132,3,4,64)
        self.circle_mapper  = FCBlock(132,3,3,64)

    def forward(self,data):
        x, edge_index = data.x, data.edges
        #return x
        #print(x.shape)
        x_0 = F.celu(self.conv1(x, edge_index))
        #x = F.dropout(x,training = True)
        
        x_1 = F.celu(self.conv2(x_0, edge_index))
        #x = F.dropout(x,training = True)

        x_2 = self.conv3(x_1,edge_index)

        return F.celu(x_2)

    def build_dag_lc(self,lines,circles):
        self.line_count   = 0
        self.circle_count = 0
        self.point_count  = 0

        x = []
        
        # just don't detect whether input segments are the same for a while.
        # the version is crude, but is should work.
        for line in lines:
            flag = False # the flag for is there is at least one line the same line with the new line
            for k in self.lines:
                if not flag and same_line(self.lines[k],line):flag = True # for each line deteced, if it is a new line, add it into the diction.
            if not flag:
                self.line_count += 1;self.lines["l{}".format(self.line_count)] = line
                
                line_feature    =  self.line_mapper(torch.cat([torch.tensor(line[0]),torch.tensor(line[1])],-1).float()).unsqueeze(0)

                x.append(line_feature)

        for circle in circles:
            flag = False # the flag for is there is at least one circle the same circle with the new circle
            for k in self.circles:
                if not flag and same_circle(self.circles[k],circle):flag = True # for each circle detected, if it is a new circle, added into the diction.
            if not flag:
                self.circle_count += 1;self.circles["c{}".format(self.circle_count)] = circle

                circle_feature    =  self.circle_mapper(torch.tensor([float(a) for a in circle]).float()).unsqueeze(0)

                x.append(circle_feature)
                #self.circle_embeddings["c{}".format(self.circle_count)] = circle_feature

        d = torch.cat(x,0)

        edges_list = [[0,1]]
        for k in self.lines:
            line = self.lines[k]
        for k in self.circles:
            circle = self.circles[k]

        connect_edges = torch.tensor(edges_list,dtype = torch.long)
        connect_edges = connect_edges.t().contiguous()

        return Data(x = d,edges = connect_edges)

    def realize_lc(self,data):
        output = self.forward(data)

        lines = [[],output[:self.line_count]];circles = [[],output[self.line_count:]]
        for i in range(self.line_count):
            lines[0].append("l{}".format(i + 1));
            #lines[1].append(output[i:i+1])
        for i in range(self.circle_count):
            circles[0].append("c{}".format(i + 1));
        return lines,circles


"""
edge_index = torch.tensor([[0, 1],
                           [1, 0],
                           [1, 2],
                           [2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index.t().contiguous())
"""