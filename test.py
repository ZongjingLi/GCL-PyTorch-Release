from config import *

from primitives import *
from datasets   import *

import matplotlib.pyplot as plt

dataset = GeometricElementsData("train","angle")

trainloader = DataLoader(dataset)

model = GeometricConstructor(model_opt)

for epoch in range(1000):
    for sample in trainloader:
        image   = sample["image"]
        concept = [term[0] for term in sample["programs"]]
      
        outputs = model.train(image,concept)

        plt.figure("inputs vs recons")
        plt.subplot(121);plt.imshow(image[0].permute([1,2,0]),cmap = "binary")
        plt.subplot(122);plt.imshow(outputs.detach())
        plt.pause(1)