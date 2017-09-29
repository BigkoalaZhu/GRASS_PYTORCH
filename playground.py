import torch
from torch import nn
from torch.autograd import Variable
from model import GRASSEncoder
from model import GRASSDecoder
from dataset import GRASS
import torch.utils.data
import util
import matplotlib.pyplot as plt
from time import time

def mse_loss(input1, target1, input2, target2):
    return (torch.sum(torch.abs(input1 - target1))/36 + torch.sum(torch.abs(input2 - target2))/16)/input1.size()[0]

config = util.get_args()

ddd = GRASS('data')
dataloader = torch.utils.data.DataLoader(ddd, batch_size=128, shuffle=True)
MSECriterion = nn.BCELoss()

model = GRASSEncoder(config)
model2 = GRASSDecoder(config)

optimizer1 = torch.optim.Adam(model.parameters())
optimizer2 = torch.optim.Adam(model2.parameters())

model.make_cuda()
model2.cuda()
MSECriterion.cuda()

plt.ion()   # something about continuous plotting

errs = []
for epoch in range(10):
    for i, data in enumerate(dataloader):
        data[0] = Variable(data[0], requires_grad = True).cuda()
        data[1] = Variable(data[1], requires_grad = True).cuda()
        data[2] = Variable(data[2], requires_grad = True).cuda()
        
        aaa = model(inputStacks=data[0], symmetryStacks=data[2], operations=data[1])

        bbb, ccc = model2(aaa, operations=data[1])
        err = mse_loss(data[0],bbb,data[2],ccc)
        #err = MSECriterion(data[0], bbb)
        model.zero_grad()
        model2.zero_grad()
        err.backward()
        optimizer1.step()
        optimizer2.step()

        errs.append(err.data[0])
        if i % 5 == 0 :
            plt.plot(errs, c='#4AD631')
            plt.draw()
            plt.pause(0.01)