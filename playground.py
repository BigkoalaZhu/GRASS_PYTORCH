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
from draw3dOBB import tryPlot
from draw3dOBB import showGenshape
from math import sqrt

def mse_loss(inputs, targets, weights):
    result = (inputs - targets)**2
    result = result.sum(2)
    result = result.sum(1)*0.4
    result = result.sum(0)/inputs.size()[0]
    return result

def mse_list_loss(inputs, targets, inSym, outSym):
    err_sum = 0
    target_list = [list(torch.split(b.squeeze(0), 1, 0)) for b in torch.split(targets, 1, 0)]
    for i in range(len(inputs)):
        for j in range(len(inputs[i])):
            err_sum = err_sum + torch.sum((inputs[i][j] - target_list[i][j])**2)
    err_sum = err_sum/len(inputs)*0.4
    return err_sum
'''
    sym_sum = 0
    sym_list = [list(torch.split(b.squeeze(0), 1, 0)) for b in torch.split(outSym, 1, 0)]
    for i in range(len(inSym)):
        for j in range(len(inSym[i])):
            sym_sum = sym_sum + torch.sum((inSym[i][j] - outSym[i][j])**2)
    sym_sum = sym_sum/len(inputs)*0.5
    return err_sum+sym_sum
'''

def encoder_weights_init(m):
    r = sqrt(6)/sqrt(2*config.hiddenSize+1)
    classname = m.__class__.__name__
    if classname.find('BoxEncoder') != -1:
        m.encoder.weight.data.uniform_(-r,r)
        m.encoder.bias.data.fill_(0)
    if classname.find('AdjEncoder') != -1:
        m.left.weight.data.uniform_(-r,r)
        m.left.bias.data.fill_(0)
        m.right.weight.data.uniform_(-r,r)
        m.second.weight.data.uniform_(-r,r)
        m.second.bias.data.fill_(0)
    if classname.find('SymEncoder') != -1:
        m.left.weight.data.uniform_(-r,r)
        m.left.bias.data.fill_(0)
        m.right.weight.data.uniform_(-r,r)
        m.right.bias.data.fill_(0)
        m.second.weight.data.uniform_(-r,r)
        m.second.bias.data.fill_(0)

def decoder_weights_init(m):
    r = sqrt(6)/sqrt(2*config.hiddenSize+1)
    classname = m.__class__.__name__
    if classname.find('BoxDecoder') != -1:
        m.decode.weight.data.uniform_(-r,r)
        m.decode.bias.data.fill_(0)
    if classname.find('AdjDecoder') != -1:
        m.left.weight.data.uniform_(-r,r)
        m.left.bias.data.fill_(0)
        m.right.weight.data.uniform_(-r,r)
        m.right.bias.data.fill_(0)
        m.decode.weight.data.uniform_(-r,r)
        m.decode.bias.data.fill_(0)
    if classname.find('SymDecoder') != -1:
        m.left.weight.data.uniform_(-r,r)
        m.left.bias.data.fill_(0)
        m.right.weight.data.uniform_(-r,r)
        m.right.bias.data.fill_(0)
        m.decode.weight.data.uniform_(-r,r)
        m.decode.bias.data.fill_(0)

config = util.get_args()

ddd = GRASS('data')
dataloader = torch.utils.data.DataLoader(ddd, batch_size=123, shuffle=True)

model = GRASSEncoder(config)
model2 = GRASSDecoder(config)

model.apply(encoder_weights_init)
model2.apply(decoder_weights_init)

optimizer1 = torch.optim.SGD(model.parameters(), lr=5e-2)
optimizer2 = torch.optim.SGD(model2.parameters(), lr=5e-2)

#model.make_cuda()
#model2.cuda()

MSECriterion = nn.MSELoss()


encoder = torch.load('encoder.pkl')
decoder = torch.load('decoder.pkl')

for i, data in enumerate(dataloader):
        data[0] = Variable(data[0])
        data[1] = Variable(data[1])
        data[2] = Variable(data[2])

        ds = torch.split(data[0], 1, 0)
        showGenshape(ds[0].squeeze(0).data.cpu().numpy())
        
        aaa = encoder(inputStacks=data[0], symmetryStacks=data[2], operations=data[1])
        bbb, ccc = decoder(aaa, operations=data[1])

        #ds = torch.split(bbb, 1, 0)
        showGenshape(torch.cat(bbb[0],0).data.cpu().numpy())
        err = mse_list_loss(bbb, data[0], ccc, data[2])


errs = []
for epoch in range(500):
    if epoch == 60:
        for param_group in optimizer1.param_groups:
            param_group['lr'] = param_group['lr']*0.33
        for param_group in optimizer2.param_groups:
            param_group['lr'] = param_group['lr']*0.33
    if epoch == 80:
        for param_group in optimizer1.param_groups:
            param_group['lr'] = param_group['lr']*0.33
        for param_group in optimizer2.param_groups:
            param_group['lr'] = param_group['lr']*0.33
    for i, data in enumerate(dataloader):
        data[0] = Variable(data[0])
        data[1] = Variable(data[1])
        data[2] = Variable(data[2])
        data[3] = Variable(data[3]).squeeze(1).squeeze(1)
        
        aaa = model(inputStacks=data[0], symmetryStacks=data[2], operations=data[1])
        bbb, ccc = model2(aaa, operations=data[1])
        err = mse_list_loss(bbb, data[0], ccc, data[2])
        #err = mse_loss(bbb, data[0], data[3])
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

torch.save(model, 'encoder.pkl')
torch.save(model2, 'decoder.pkl')
