import argparse
import torch

parser = argparse.ArgumentParser()

parser.add_argument("--encoder_dim",default = 128, type = str, help = "the encoder output dim")
parser.add_argument("--latent_dim",default = 32//2, type = int, help = "the latent propagation dimension for the dag construction")
parser.add_argument("--resolution",default = (64,64), type = tuple, help = "the default resolution configuration")

model_opt = parser.parse_args(args = [])