import torch
import torch.nn as nn

from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing

from config import *

def same_circle(a,b):return False

def same_line(a,b):return False

class LCNet(MessagePassing):
    def __init__(self,opt = model_opt):
        super().__init__()

        # record features from the input lines and circles
        self.line_features = {}
        self.circle_features = {}
        # after the propagation

        # construct the crude rpresentation of input lines and circles
        self.lines = {}
        self.circles = {}

        # the name of the points will be generated automatically

    def forward(self,x):return 0

    def build_lc(self,lines,circles):
        self.line_count   = 0
        self.circle_count = 0
        for line in lines:
            flag = True
            for k in self.lines:
                if same_line(self.lines[k],line) and flag:flag = False
            if flag:
                self.line_count += 1
                self.lines["p{}".format(self.line_count)] = line
        for circle in circles:
            flag = True
            for k in self.circles:
                if same_circle(self.circles[k],circle) and flag:flag = False
            if flag:
                self.circle_count += 1
                self.lines["p{}".format(self.circle_count)] = circle
        return 0

    def realize_lc(self):
        return 0