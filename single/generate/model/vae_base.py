#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
Created on Jun 23, 2021
@author: nakaizura
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


# VAE model
# 输入，建模分布的mu和var，采样得到向量，然后重建+KL约束
class VAE(nn.Module):
    def __init__(self, encode_dims=[512, 20], decode_dims=[20, 512], dropout=0.0):

        super(VAE, self).__init__()
        #这里应该接入别的

        #从这里开始
        '''
        平均值和方差层为
        (fc_mu): Linear(in_features=512, out_features=20, bias=True)
        (fc_logvar): Linear(in_features=512, out_features=20, bias=True)
        这意味着隐层是20维
        '''
        self.fc_mu = nn.Linear(encode_dims[-2], encode_dims[-1])  # 学习mu和var
        self.fc_logvar = nn.Linear(encode_dims[-2], encode_dims[-1])

        '''
        (decoder): 
        (dec_0): Linear(in_features=20, out_features=128, bias=True)
        (dec_1): Linear(in_features=128, out_features=768, bias=True)
        (dec_2): Linear(in_features=768, out_features=1024, bias=True)
        '''
        self.decoder = nn.ModuleDict({
            f'dec_{i}': nn.Linear(decode_dims[i], decode_dims[i + 1])
            for i in range(len(decode_dims) - 1)
        })

        #隐层维度
        self.latent_dim = encode_dims[-1]

        #防止过拟合
        self.dropout = nn.Dropout(p=dropout)

        #为了转换为主题向量
        self.fc1 = nn.Linear(encode_dims[-1], encode_dims[-1])

    def encode(self, x):  # 编码，但是在整体模型中去掉多层感知器
        hid = x#传入的编码应该是256维度
        #hid是LSTM编码之后的256维度向量
        #for i, layer in self.encoder.items():  # 多层fc，不需要
        #    hid = F.relu(self.dropout(layer(hid)))
        mu, log_var = self.fc_mu(hid), self.fc_logvar(hid)  # 得到mu和var
        return mu, log_var

    def inference(self, x):  # 推断
        mu, log_var = self.encode(x)  # 得到分布
        theta = torch.softmax(x, dim=1)  # 得到向量
        return theta

    def reparameterize(self, mu, log_var):  # 重参数技巧，使训练可微
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)  # 采样
        z = mu + eps * std
        return z

    def decode(self, z):  # 解码
        hid = z
        for i, (_, layer) in enumerate(self.decoder.items()):  # 多层fc
            hid = layer(hid)
            if i < len(self.decoder) - 1:
                hid = F.relu(self.dropout(hid))
        return hid

    def forward(self, x):
        mu, log_var = self.encode(x)  # 得到分布的mu和var
        _theta1 = self.reparameterize(mu, log_var)  # 重参数采样得到向量
        _theta2 = self.fc1(_theta1)
        x_reconst = self.decode(_theta2)  # 重建loss
        return x_reconst, mu, log_var , _theta2  # 返回重建和两个分布参数，KL散度在模型中计算，不在此处


if __name__ == '__main__':
    model = VAE(encode_dims=[1024, 512, 256, 20], decode_dims=[20, 128, 768, 1024])
    print(model)
    model = model.cuda()
    inpt = torch.randn(234, 1024).cuda()
    out, mu, log_var = model(inpt)
    print(out.shape)
    print(mu.shape)