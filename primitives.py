import torch
import torch.nn as nn

from moic.data_structure import *
from moic.mklearn.nn.functional_net import FCBlock

from config import *

def ptype(inputs):
    if inputs[0] == "c": return "circle"
    if inputs[0] == "l": return "line" 
    if inputs[0] == "p": return "point"

# geometric structure model. This is used to create the geometric concept graph
# and realize the concept and make sample of concepts. This is practically the decoder part of the 
# Geometric AutoEncoder model

dgc = ["l1 = line(p1(), p2())","c1* = circle(p1(), p2())","c2* = circle(p2(), p1())","l2 = line(p1(), p3(c1, c2))","l3 = line(p2(), p3()))"]

def parse_geoclidean(programs = dgc):
    outputs = []
    for program in programs:
        left,right = program.split("=")
        left = left.replace(" ","");right = right.replace(" ","")
        func_node_form = toFuncNode(right)
        func_node_form.token = left
        outputs.append(func_node_form)
    return outputs

class MessageProp(nn.Module):
    def __init__(self,opt):
        super().__init__()
        self.update_map   = nn.Linear(opt.geometric_latent_dim,opt.geometric_latent_dim)
        self.message_map  = nn.Linear(opt.geometric_latent_dim,opt.geometric_latent_dim)
        self.joint_update = FCBlock(132,2,2 *  opt.geometric_latent_dim,opt.geometric_latent_dim)
        self.opt = opt

    def forward(self,signal,components):
        if not components: 
            return self.joint_update(torch.cat([signal,torch.zeros([1,self.opt.geometric_latent_dim])] ,-1))
        right_inters = 0
        for comp in components:right_inters += self.message_map(comp)

        right_inters = self.update_map(right_inters)
        return self.joint_update(torch.cat([signal,right_inters],-1))

def find_connection(node,graph,loc = 0):
    outputs = []
    for edge in graph.edges:
        if edge[loc] == node:outputs.append(edge[int(not loc)])
    return outputs

class GeometricConstructor(nn.Module):
    def __init__(self,opt = model_opt):
        super().__init__()