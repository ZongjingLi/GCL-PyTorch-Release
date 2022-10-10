import torch
import torch.nn as nn

import argparse

train_parser = argparse.ArgumentParser()
train_parser.add_argument("--epoch", default = 9999, type = int, help = "the default trainingepoch for the model")
train_parser.add_argument("--lr", default = 2e-4, type = float, help = "the learning rate for the optimizer")
train_parser.add_argument("--mode", default = "Mode", type = str, help = "the training mode for the geometric concept")

train_config = train_parser.parse_args(args = [])