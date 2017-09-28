import math
import torch
from torch import nn
from torch.autograd import Variable

import itertools

def bundle(input):
    if input is None:
        return None
    input = tuple(input)
    return torch.cat(input, 0)

def unbundle(input):
    if input is None:
        return itertools.repeat(None)
    return torch.split(input, 1, 0)

class BoxEncoder(nn.Module):

    def __init__(self, boxSize, featureSize):
        super(BoxEncoder, self).__init__()
        self.encoder = nn.Linear(boxSize, featureSize)
        self.tanh = nn.Tanh()

    def forward(self, boxes_in):
        boxes = bundle(boxes_in)
        boxes = self.encoder(boxes)
        boxes = self.tanh(boxes)
        return unbundle(boxes)

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
    
    def forward(self, inputStacks, symmetryStacks, operations):
        buffers = []
        boxes = [list(torch.split(b.squeeze(1), 1, 0)) for b in torch.split(inputStacks, 1, 1)]
        for b in boxes:
            buffers.append(self.boxEncoder(b))
        symBuffers = [list(torch.split(b.squeeze(1), 1, 0)) for b in torch.split(symmetryStacks, 1, 1)]
        stacks = [[buf[0], buf[0]] for buf in buffers]
        symStacks = [[buf[0], buf[0]] for buf in symBuffers]
        num_operations = operations.size(0)
        for i in range(num_operations):
            if operations is not None:
                opt = operations[i]
            lefts, rights, features, syms = [], [], [], []
            batch = zip(opt.data, buffers, stacks, symBuffers, symStacks)
            for op, buf, stack, sBuf, sStack in batch:              
                if op == 0:
                    stack.append(buf.pop())
                if op == 1:
                    stack.append(buf.pop())
                    sStack.append(sBuf.pop())
                if op == 2:
                    rights.append(stack.pop())
                    lefts.append(stack.pop())
                if op == 3:
                    features.append(stack.pop())
                    syms.append(sStack.pop())
            
            if lefts:
                reduced = iter(self.adjEncoder(lefts, rights))
                for op, stack in zip(opt.data, stacks):
                    if op == 2:
                        stack.append(next(reduced))
            
            if features:
                reduced = iter(self.symEncoder(features, syms))
                for op, stack in zip(opt.data, stacks):
                    if op == 3:
                        stack.append(next(reduced))
        return bundle([stack.pop() for stack in stacks])[0]

class GRASSDecoder(nn.Module):
    def __init__(self, config):
        super(GRASSDecoder, self).__init__()
        self.boxDecoder = BoxDecoder(boxSize = config.boxSize, featureSize = config.featureSize)
        self.adjDecoder = AdjDecoder(featureSize = config.featureSize, hiddenSize = config.hiddenSize)
        self.symDecoder = SymDecoder(featureSize = config.featureSize, symmetrySize = config.symmetrySize, hiddenSize = config.hiddenSize)

    def forward(self, inputStacks, operations):
        features = [list(torch.split(b.squeeze(1), 1, 0)) for b in torch.split(inputStacks, 1, 1)]
        if operations is not None:
            stacks = [[buf[0]] for buf in features]
            symStacks = [[] for buf in features]
            num_operations = operations.size(0)
            for i in range(num_operations):
                if operations is not None:
                    opt = operations[i]
                proximityD, symmetryD = [], []
                batch = zip(opt.data, stacks)
                for op, stack in batch:
                    if op == 2:
                        proximityD.append(stack.pop())
                    if op == 3:
                        symmetryD.append(stack.pop())
                if proximityD:
                    lefts, rights = iter(self.adjDecoder(proximityD))
                    for op, stack in zip(opt.data, stacks):
                        if op == 2:
                            stack.append(next(lefts))
                            stack.append(next(rights))
                if symmetryD:
                    fs, ss = iter(self.symDecoder(symmetryD))
                    for op, stack, sStack in zip(opt.data, stacks, symStacks):
                        if op == 3:
                            stack.append(next(fs))
                            sStack.append(next(ss))
            stacks = self.boxDecoder(stacks)
            return bundle(stacks), bundle(symStacks)