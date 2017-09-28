import os
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description='PyTorch/GRASS')
    parser.add_argument('--boxSize', type=int, default=12)
    parser.add_argument('--featureSize', type=int, default=80)
    args = parser.parse_args()
    return args