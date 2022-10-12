import torch
import torch.nn as nn

from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing

from config import *

class LCNet(MessagePassing):
    def __init__(self,opt = model_opt):
        super().__init__()

    def forward(self,x):return 0