
import torch
import torch.nn as nn

import argparse

from datasets import *
from detection import *
from primitives import *
from lcnet import *
from model import *
from config import *

torch.autograd.set_detect_anomaly(True)

train_parser = argparse.ArgumentParser()
train_parser.add_argument("--epoch", default = 300, type = int, help = "the default trainingepoch for the model")
train_parser.add_argument("--lr", default = 2e-4, type = float, help = "the learning rate for the optimizer")
train_parser.add_argument("--mode", default = "Mode", type = str, help = "the training mode for the geometric concept")
train_parser.add_argument("--checkpoint_path", default = None, help = "the location of the check point")
train_config = train_parser.parse_args(args = [])

# Binary Cross Entropy Loss, the criterion for the REINFORCE
BCELoss = torch.nn.BCELoss(reduction = "mean")

visualize = True

model = GCL(model_opt)
model = torch.load("model.ckpt")

dataset = GeometricElementsData("train","angle") # the geometric concept dataset for training
dataset = AllElementsData("train")
train_loader = DataLoader(dataset, batch_size = 1, shuffle = True)

# the Adam optimizer for the constuctor and lcnet
con_optimizer = torch.optim.Adam(model.parameters(), lr = 2e-4);plt.ion()

bce_history = []
for epoch in range(train_config.epoch):
    
    # clear the gradient and reset the loss
    total_loss = 0
    bce = 0;lp =0;
    for sample in train_loader:
        raw_concept,image,path = sample["concept"],sample["image"],sample["path"]
        #print(raw_concept,path)
        concept = [t[0] for t in raw_concept]

        recons,logp = model(concept,image,path)
        
        bce_loss = BCELoss(torch.tensor(recons).float()/256,image[0][0].float())
        total_loss += logp * bce_loss.log()
        bce += bce_loss;lp+=logp

    bce_history.append(bce)
    if epoch % 1 == 0:
        print("epoch:{} Logp:{} BCE:{}".format(epoch,lp,bce))
        plt.figure("loss");plt.cla()
        plt.plot(bce_history);plt.pause(1)
        torch.save(model,"model.ckpt")
    # calculate all the gradient and do a REINFORCE
    con_optimizer.zero_grad()
    total_loss.backward(retain_graph = True)
    con_optimizer.step();
    
    

