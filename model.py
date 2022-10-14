import torch
import torch.nn as nn

from primitives import *
from lcnet import *
from detection import *

class GCL(nn.Module):
    def __init__(self,model_opt):
        super().__init__()
        self.constructor = GeometricConstructor(model_opt) # the geometric concept encoder 
        self.lcnet = LCNet(model_opt) # the encoder of lines and circles    

    def forward(self,concept,image,path,visualize = True):
        constructor = self.constructor
        lcnet = self.lcnet
        # realize the concept using the geometric constructor
        constructor.build_dag(concept)
        constructor.realize(torch.zeros([1,128]))        
        # detect visible components (line and circles) from the image

        lines,circles = detect_lines_and_circles(path[0])

        data = lcnet.build_dag_lc(lines,circles) # use the lc net to create the connection graph
        lines,circles = lcnet.realize_lc(data) # propgate to get the embedding according to the relations.

        # the reconstruction and the logp of that reconstruction
        recons,logp =   constructor.construct(lines,circles,lcnet.lines,lcnet.circles)
        recons = recons[:,:,0]
        print(recons.shape)
        if visualize:
            plt.figure("example");plt.subplot(121);plt.cla();plt.imshow(image[0][0],cmap = "bone")
            plt.subplot(122);plt.cla();plt.imshow(recons,cmap = "bone")
            #plt.figure("concept");plt.cla();nx.draw_networkx(constructor.structure)

        return recons,logp