'''
Created on Jun 23, 2021
@author: nakaizura
'''
import os
import re
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from model.vae_base import VAE
import matplotlib.pyplot as plt
import sys
import codecs
import time

sys.path.append('..')
# from utils import evaluate_topic_quality, smooth_curve


class GSM:
    def __init__(self , config):
        self.config = config
        self.input_dim = config.input_dim  # bow维度，输入维度
        self.n_topic = config.topic  # 主题数，隐层维度
        # TBD_fc1
        self.vae = VAE(encode_dims=[self.input_dim, self.n_topic], decode_dims=[self.n_topic, 512, self.input_dim],
                       dropout=0.0)  # VAE


    def forward(self , input):
        return self.vae(input)



if __name__ == '__main__':
    model = VAE(encode_dims=[1024, 512, 256, 20], decode_dims=[20, 128, 768, 1024])
    model = model.cuda()
    inpt = torch.randn(234, 1024).cuda()
    out, mu, log_var = model(inpt)
    print(out.shape)
    print(mu.shape)