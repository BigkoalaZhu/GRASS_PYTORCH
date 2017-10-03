import math
import torch
from torch import nn
from torch.autograd import Variable
from time import time

import itertools

def bundleComplete(input, num):
    if input is None:
        return None
    a = input[0].size()[0]
    b = input[0].size()[1]
    if len(input) < num:
        for i in range(num-len(input)):
            input.append(Variable(torch.zeros(a,b)))
    input = tuple(input)
    return torch.unsqueeze(torch.cat(input, 0), 0)

def bundle(input):
    if input is None:
        return None
    input = tuple(input)
    return torch.cat(input, 0)

def unbundle(input):
    if input is None:
        return itertools.repeat(None)
    return list(torch.split(input, 1, 0))

class BoxEncoder(nn.Module):

    def __init__(self, boxSize, featureSize):
        super(BoxEncoder, self).__init__()
        self.encoder = nn.Linear(boxSize, featureSize)
        self.tanh = nn.Tanh()

    def forward(self, boxes_in):
        #boxes = bundle(boxes_in)
        boxes = self.encoder(boxes_in)
        boxes = self.tanh(boxes)
        #return unbundle(boxes)
        return boxes

class AdjEncoder(nn.Module):

    def __init__(self, featureSize, hiddenSize):
        super(AdjEncoder, self).__init__()
        self.left = nn.Linear(featureSize, hiddenSize)
        self.right = nn.Linear(featureSize, hiddenSize, bias=False)
        self.second = nn.Linear(hiddenSize, featureSize)
        self.tanh = nn.Tanh()

    def forward(self, left_in, right_in):
        left, right = bundle(left_in), bundle(right_in)
        out = self.left(left)
        out += self.right(right)
        out = self.tanh(out)
        out = self.second(out)
        out = self.tanh(out)
        return unbundle(out)

class SymEncoder(nn.Module):

    def __init__(self, featureSize, symmetrySize, hiddenSize):
        super(SymEncoder, self).__init__()
        self.left = nn.Linear(featureSize, hiddenSize)
        self.right = nn.Linear(symmetrySize, hiddenSize)
        self.second = nn.Linear(hiddenSize, featureSize)
        self.tanh = nn.Tanh()

    def forward(self, left_in, right_in):
        left, right = bundle(left_in), bundle(right_in)
        out = self.left(left)
        out += self.right(right)
        out = self.tanh(out)
        out = self.second(out)
        out = self.tanh(out)
        return unbundle(out)

class AdjDecoder(nn.Module):

    def __init__(self, featureSize, hiddenSize):
        super(AdjDecoder, self).__init__()
        self.decode = nn.Linear(featureSize, hiddenSize)
        self.left = nn.Linear(hiddenSize, featureSize)
        self.right = nn.Linear(hiddenSize, featureSize)
        self.tanh = nn.Tanh()

    def forward(self, parent_in):
        parent = bundle(parent_in)
        out = self.decode(parent)
        out = self.tanh(out)
        l = self.left(out)
        r = self.right(out)
        l = self.tanh(l)
        r = self.tanh(r)
        return unbundle(l), unbundle(r)

class SymDecoder(nn.Module):

    def __init__(self, featureSize, symmetrySize, hiddenSize):
        super(SymDecoder, self).__init__()
        self.decode = nn.Linear(featureSize, hiddenSize)
        self.tanh = nn.Tanh()
        self.left = nn.Linear(hiddenSize, featureSize)
        self.right = nn.Linear(hiddenSize, symmetrySize)

    def forward(self, parent_in):
        parent = bundle(parent_in)
        out = self.decode(parent)
        out = self.tanh(out)
        f = self.left(out)
        f = self.tanh(f)
        s = self.right(out)
        s = self.tanh(s)
        return unbundle(f), unbundle(s)

class BoxDecoder(nn.Module):

    def __init__(self, boxSize, featureSize):
        super(BoxDecoder, self).__init__()
        self.decode = nn.Linear(featureSize, boxSize)
        self.tanh = nn.Tanh()

    def forward(self, parent_in):
        parent = bundle(parent_in)
        out = self.decode(parent)
        out = self.tanh(out)
        return unbundle(out)

class GRASSEncoder(nn.Module):
    def __init__(self, config):
        super(GRASSEncoder, self).__init__()
        self.boxEncoder = BoxEncoder(boxSize = config.boxSize, featureSize = config.featureSize)
        self.adjEncoder = AdjEncoder(featureSize = config.featureSize, hiddenSize = config.hiddenSize)
        self.symEncoder = SymEncoder(featureSize = config.featureSize, symmetrySize = config.symmetrySize, hiddenSize = config.hiddenSize)
    
    def make_cuda(self):
        self.boxEncoder.cuda()
        self.adjEncoder.cuda()
        self.symEncoder.cuda()

    def forward(self, inputStacks, symmetryStacks, operations):
        buffers = []
        encoded = self.boxEncoder(inputStacks)
        buffers = [list(torch.split(b.squeeze(0), 1, 0)) for b in torch.split(encoded, 1, 0)]
        symBuffers = [list(torch.split(b.squeeze(0), 1, 0)) for b in torch.split(symmetryStacks, 1, 0)]
        stacks = [[] for buf in buffers]
        operations = torch.t(operations.squeeze(1))
        num_operations = operations.size(0)
        for i in range(num_operations):
            if operations is not None:
                opt = operations[i]
            lefts, rights, features, syms = [], [], [], []
            batch = zip(opt.data, buffers, stacks, symBuffers)

            for op, buf, stack, sBuf in batch:              
                if op == 0:
                    stack.append(buf.pop())
                if op == 1:
                    rights.append(stack.pop())
                    lefts.append(stack.pop())
                if op == 2:
                    features.append(stack.pop())
                    syms.append(sBuf.pop())              
            
            if lefts:
                reduced = iter(self.adjEncoder(lefts, rights))
                for op, stack in zip(opt.data, stacks):
                    if op == 1:
                        stack.append(next(reduced))
            
            if features:
                reduced = iter(self.symEncoder(features, syms))
                for op, stack in zip(opt.data, stacks):
                    if op == 2:
                        stack.append(next(reduced))
        return bundle([stack.pop() for stack in stacks])

class GRASSDecoder(nn.Module):
    def __init__(self, config):
        super(GRASSDecoder, self).__init__()
        self.boxDecoder = BoxDecoder(boxSize = config.boxSize, featureSize = config.featureSize)
        self.adjDecoder = AdjDecoder(featureSize = config.featureSize, hiddenSize = config.hiddenSize)
        self.symDecoder = SymDecoder(featureSize = config.featureSize, symmetrySize = config.symmetrySize, hiddenSize = config.hiddenSize)
        self.maxBoxes = config.maxBoxes
        self.maxSyms = config.maxSyms
        self.symmetrySize = config.symmetrySize

    def wholeTree(self, inputStacks, operations):
        features = [b for b in torch.split(inputStacks, 1, 0)]
        if operations is not None:
            stacks = [[buf] for buf in features]
            boxStacks = [[] for buf in features]
            symStacks = [[] for buf in features]
            wholeSymStacks = [[] for buf in features]
            operations = torch.t(operations.squeeze(1))
            num_operations = operations.size(0)
            for i in range(num_operations):
                if operations is not None:
                    opt = operations[num_operations - i - 1]
                proximityD, symmetryD = [], []
                batch = zip(opt.data, stacks, boxStacks)
                for op, stack, bStack in batch:
                    if op == 0:
                        bStack.append(stack.pop())
                    if op == 1:
                        proximityD.append(stack.pop())
                    if op == 2:
                        symmetryD.append(stack.pop())
                if proximityD:
                    lefts, rights = self.adjDecoder(proximityD)
                    count = 0
                    for op, stack, wss in zip(opt.data, stacks, wholeSymStacks):
                        if op == 1:
                            stack.append(lefts[count])
                            stack.append(rights[count])
                            if len(wss) == 0:
                                wss.append(Variable(torch.zeros(self.symmetrySize).cuda()))
                                wss.append(Variable(torch.zeros(self.symmetrySize).cuda()))
                            else:
                                wss.append(wss[len(wss)-1])
                                wss.append(wss[len(wss)-1])
                            count = count + 1
                if symmetryD:
                    ff, fs = self.symDecoder(symmetryD)
                    count = 0
                    for op, stack, sStack, wss in zip(opt.data, stacks, symStacks, wholeSymStacks):
                        if op == 2:
                            stack.append(ff[count])
                            sStack.append(fs[count])
                            wss.append(fs[count].squeeze(0))
                            count = count + 1
            
            return torch.cat(boxStacks), torch.cat(symStacks)
        
    def forward(self, inputStacks, operations):
        features = [b for b in torch.split(inputStacks, 1, 0)]
        if operations is not None:
            stacks = [[buf] for buf in features]
            boxStacks = [[] for buf in features]
            symStacks = [[] for buf in features]
            operations = torch.t(operations.squeeze(1))
            num_operations = operations.size(0)
            for i in range(num_operations):
                if operations is not None:
                    opt = operations[num_operations - i - 1]
                proximityD, symmetryD = [], []
                batch = zip(opt.data, stacks, boxStacks)
                for op, stack, bStack in batch:
                    if op == 0:
                        bStack.append(stack.pop())
                    if op == 1:
                        proximityD.append(stack.pop())
                    if op == 2:
                        symmetryD.append(stack.pop())
                if proximityD:
                    lefts, rights = self.adjDecoder(proximityD)
                    count = 0
                    for op, stack in zip(opt.data, stacks):
                        if op == 1:
                            stack.append(lefts[count])
                            stack.append(rights[count])
                            count = count + 1
                if symmetryD:
                    ff, fs = self.symDecoder(symmetryD)
                    count = 0
                    for op, stack, sStack in zip(opt.data, stacks, symStacks):
                        if op == 2:
                            stack.append(ff[count])
                            sStack.append(fs[count])
                            count = count + 1
            for i in range(len(boxStacks)):
                boxStacks[i].reverse()
                symStacks[i].reverse()
                boxStacks[i] = bundleComplete(self.boxDecoder(boxStacks[i]), self.maxBoxes)
                symStacks[i] = bundleComplete((symStacks[i]), self.maxSyms)
            return torch.cat(boxStacks), torch.cat(symStacks)