#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 13:04:34 2018

@author: manoj
"""

import torch
import torch.nn as nn

class RN(nn.Module):
    def __init__(self,Ncls,**kwargs):
        super().__init__()

        I_CNN = 2048
        Q_GRU_out = 1024
        Q_embedding = 300
        self.Ncls = Ncls
        self.QRNN = nn.LSTM(Q_embedding,Q_GRU_out,num_layers=1,bidirectional=False)
                
        hidden = 512
        insize = I_CNN + Q_GRU_out
        self.W = nn.Linear(insize,hidden)
        self.Wprime = nn.Linear(insize,hidden)      
        self.f = nn.Linear(in_features=hidden,out_features=1,bias=False)
        

    def forward(self,wholefeat,pooled,box_feats,q_feats,box_coords,index):

        enc2,_ = self.QRNN(q_feats.permute(1,0,2))
        q_rnn = enc2[-1]       
        b,d,k = box_feats.size()
        qst  =  q_rnn.unsqueeze(1)
        qst = qst.repeat(1, d, 1)        
        b_full = torch.cat([qst,box_feats],-1)            
        #gated tanh function
        y_tilde = torch.tanh(self.W(b_full))
        g = torch.sigmoid(self.Wprime(b_full))
        si = torch.mul(y_tilde, g)# gating   
        wsi = torch.sigmoid(self.f(si))
        counts = wsi.sum(dim=1)    
        return counts.squeeze(1)

