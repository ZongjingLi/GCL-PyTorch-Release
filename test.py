from config import *

from primitives import *
from datasets   import *

import matplotlib.pyplot as plt

dataset = GCLData("r1")

trainloader = DataLoader(dataset)

model = GeometricConstructor(model_opt)

optim = torch.optim.Adam(model.parameters(), lr = 1e-3)

r1 = [
    'c1 = circle(p1(), p2())',
    'c2 = circle(p3(c1), p4())',
    'l3 = line(p5(c1), p6(c1, c2))'
    ]

for epoch in range(1000):
    loss = 0
    optim.zero_grad()
    for sample in trainloader:
        
        image   = sample["image"]
        #concept = [term[0] for term in sample["programs"]]
        concept = r1
      
        outputs = model.train(image,concept)

        plt.figure("inputs vs recons")
        plt.subplot(121);plt.imshow(image[0].permute([1,2,0]),cmap = "binary")
        plt.subplot(122);plt.imshow(outputs[0].detach().exp().permute([1,2,0]),cmap = "binary")
        plt.pause(0.01)

        loss += model.ploss
    loss.backward()
    optim.step()
    print(loss)