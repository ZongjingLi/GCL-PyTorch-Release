import torch
import torch.nn as nn

import argparse

from datasets import *
from detection import *
from primitives import *
from lcnet import *
from config import *

train_parser = argparse.ArgumentParser()
train_parser.add_argument("--epoch", default = 9999, type = int, help = "the default trainingepoch for the model")
train_parser.add_argument("--lr", default = 2e-4, type = float, help = "the learning rate for the optimizer")
train_parser.add_argument("--mode", default = "Mode", type = str, help = "the training mode for the geometric concept")
train_parser.add_argument("--checkpoint_path", default = None, help = "the location of the check point")
train_config = train_parser.parse_args(args = [])

# Binary Cross Entropy Loss, the criterion for the REINFORCE
BCELoss = torch.nn.BCELoss(reduction = "mean")

visualize = True

constructor = GeometricConstructor() # the geometric concept encoder 
lcnet = LCNet() # the encoder of lines and circles

dataset = GeometricElementsData("train","angle") # the geometric concept dataset for training
train_loader = DataLoader(dataset, batch_size = 1, shuffle = True)

# the Adam optimizer for the constuctor and lcnet
con_optimizer = torch.optim.Adam(constructor.parameters(), lr = 2e-4)
lc_optimizer  = torch.optim.Adam(lcnet.parameters(), lr = 2e-4)

for epoch in range(train_config.epoch):
    total_loss = 0
    # clear the gradient and reset the loss
    con_optimizer.zero_grad()
    lc_optimizer.zero_grad()

    for sample in train_loader:
        raw_concept,image = sample["concept"],sample["image"]
        
        concept = [t[0] for t in raw_concept]
        # realize the concept using the geometric constructor
        constructor.build_dag(concept)
        constructor.realize(torch.zeros([1,128]))

        if visualize:
            plt.figure("example");plt.cla();plt.imshow(image[0][0],cmap = "binary")
            plt.figure("concept");plt.cla();nx.draw_networkx(constructor.structure)
        plt.show()
        # detect visible components (line and circles) from the image
        lines,circles = detect_lines_and_circles(image)
        lcnet.build_dag_lc(lines,circles) # use the lc net to create the connection graph
        lcnet.realize_lc() # propgate to get the embedding according to the relations.

        # the reconstruction and the logp of that reconstruction
        recons,logp =   model.construct(lcnet)
        total_loss  +=  logp * BCELoss(recons,image) # the reinforce loss, the prob of a recon and the loss of that recon.

    # calculate all the gradient and do a REINFORCE
    total_loss.backward()
    con_optimizer.step()
    lc_optimizer.step()
