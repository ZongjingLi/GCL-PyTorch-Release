import torch
import torch.nn as nn

from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing

from config import *

class LCNet(MessagePassing):
    def __init__(self,opt = model_opt):
        super().__init__()

        # record features from the input lines and circles
        self.line_features = {}
        self.circle_features = {}
        # after the propagation

        # construct the crude rpresentation of input lines and circles
        self.line = {}
        self.circles = {}

        # the name of the points will be generated automatically

    def forward(self,x):return 0

    def build_lc(self,lines,circles):
        return 0

    def realize_lc(self):
        return 0