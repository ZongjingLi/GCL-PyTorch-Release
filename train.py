import torch
import torch.nn as nn

import argparse

from datasets import *
from detection import *
from primitives import *
from config import *

train_parser = argparse.ArgumentParser()
train_parser.add_argument("--epoch", default = 9999, type = int, help = "the default trainingepoch for the model")
train_parser.add_argument("--lr", default = 2e-4, type = float, help = "the learning rate for the optimizer")
train_parser.add_argument("--mode", default = "Mode", type = str, help = "the training mode for the geometric concept")
train_parser.add_argument("--checkpoint_path", default = None, help = "the location of the check point")
train_config = train_parser.parse_args(args = [])

BCELoss = torch.nn.BCELoss(reduction = "mean")

model = GeometricConstructor()

dataset = []
train_loader = DataLoader(dataset, batch_size = 1, shuffle = True)

optimizer = torch.optim.Adam(model.parameters(), lr = 2e-4)

for epoch in range(train_config.epoch):
    total_loss = 0
    optimizer.zero_grad()
    for sample in train_loader:
        concept,image = sample["concept"],sample["image"]

        model.build_dag(concept)
        model.realize()

        lines,circles = detect_lines_and_circles(image)

    total_loss.backward()
    optimizer.step()