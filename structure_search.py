import torch
import torch.nn as nn

import numpy as np
import networkx as nx

class ConceptSearch(nn.Module):
    def __init__(self):
        super().__init__()
        self.heuistics = None

        self.best_concepts = []

    def get_node(self,node):
        return node
    
    def search(self,image):
        return 0

if __name__ == "__main__":
    astar_search = ConceptSearch()

    