import torch
import torch.nn as nn

from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing

import networkx as nx

from config import *

def same_circle(a,b):return False

def same_line(a,b):return False

class LCNet(MessagePassing):
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
        self.points  = {}

        # the name of the points will be generated automatically
        self.graph = nx.DiGraph()

        self.net_data = 0

        self.update_output = nn.Linear(3,3)

    def forward(self,x):return 0

    def build_dag_lc(self,lines,circles):
        self.line_count   = 0
        self.circle_count = 0
        self.point_count  = 0
    
        for line in lines:
            flag = False # the flag for is there is at least one line the same line with the new line
            for k in self.lines:
                if not flag and same_line(self.lines[k],line):flag = True # for each line deteced, if it is a new line, add it into the diction.
            if not flag:self.line_count += 1;self.lines["l{}".format(self.line_count)] = line
        for circle in circles:
            flag = False # the flag for is there is at least one circle the same circle with the new circle
            for k in self.circles:
                if not flag and same_circle(self.circles[k],circle):flag = True # for each circle detected, if it is a new circle, added into the diction.
            if not flag:self.circle_count += 1;self.lines["c{}".format(self.circle_count)] = circle
    
        x = []
        edges = []
        net_data = Data(x,edges)
        return 0

    def realize_lc(self):
        x = self.net_data
        return self.forward(x)