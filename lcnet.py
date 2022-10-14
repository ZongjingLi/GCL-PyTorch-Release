import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from moic.mklearn.nn.functional_net import FCBlock

import networkx as nx

from config import *

def same_circle(a,b):return False

def same_line(a,b):return False

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
        self.conv1  =  GCNConv(64, 32)
        self.conv2  =  GCNConv(32, 64)
        self.conv3  =  GCNConv(64, 32)
        # use the graph convolution to map the data from the point and circle
        self.line_mapper    = FCBlock(132,3,4,64)
        self.circle_mapper  = FCBlock(132,3,3,64)

    def forward(self,data):
        x, edge_index = data.x, data.edges
        x = self.conv1(x, edge_index)
        x = F.celu(x)
        x = F.dropout(x,training = True)

        x = self.conv2(x, edge_index)
        x = F.celu(x)
        x = F.dropout(x,training = True)
        
        x = self.conv3(x,edge_index)

        return F.celu(x)

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
                #print(line)
                line_feature    =  self.line_mapper(torch.cat([torch.tensor(line[0]),torch.tensor(line[1])],-1).float()).unsqueeze(0)
                x.append(line_feature)
                self.line_embeddings["l{}".format(self.line_count)] = line_feature
        for circle in circles:
            flag = False # the flag for is there is at least one circle the same circle with the new circle
            for k in self.circles:
                if not flag and same_circle(self.circles[k],circle):flag = True # for each circle detected, if it is a new circle, added into the diction.
            if not flag:
                self.circle_count += 1;self.circles["c{}".format(self.circle_count)] = circle
                print(circle)
                circle_feature    =  self.circle_mapper(torch.tensor(circle).float()).unsqueeze(0)
                x.append(circle_feature)
                self.circle_embeddings["c{}".format(self.circle_count)] = circle_feature

        x = torch.cat(x,0)
        connect_edges = torch.tensor([
            [1,2],
        ],dtype = torch.long)
        connect_edges = connect_edges.t().contiguous()

        self.net_data = Data(x = x,edges = connect_edges)
        return self.net_data

    def realize_lc(self):
        x = self.net_data
        output = self.forward(x)
        for i in range(self.line_count):
            self.line_embeddings["l{}".format(i + 1)] = output[i]
        for i in range(self.circle_count):
            self.circle_embeddings["c{}".format(i + 1)] = output[i]
        
"""
edge_index = torch.tensor([[0, 1],
                           [1, 0],
                           [1, 2],
                           [2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index.t().contiguous())
"""