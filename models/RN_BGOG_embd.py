#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 13:04:34 2018

@author: manoj
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .lang_new import QuestionParser



class RN(nn.Module):
    def __init__(self,Ncls,**kwargs):
        super().__init__()

        I_CNN = 2048
        Q_GRU_out = 1024
        Q_embedding = 300
        LINsize = 1024
        Boxcoords = 4

        self.Ncls = Ncls
        #self.Ncls = 1
        
        
        if kwargs.get('trainembd'):       
            self.QRNN = QuestionParser(dictionaryfile = kwargs['dictionaryfile'],
                                       glove_file = kwargs['glove'],
                                         dropout=0.3, word_dim=Q_embedding,
                                         ques_dim=Q_GRU_out ,
                                         rnn_type= 'GRU')
               

        layers_g1 = [ nn.Linear( 2*I_CNN + 2*Boxcoords + Q_GRU_out, LINsize),
               nn.ReLU(inplace=True),
               #nn.Dropout(0.5),
               nn.Linear( LINsize, LINsize),
               nn.ReLU(inplace=True),
               #nn.Dropout(0.5),
               nn.Linear(LINsize,LINsize),
               nn.ReLU(inplace=True)]

        self.g1 = nn.Sequential(*layers_g1)

        self.f1_phi = nn.Linear(LINsize,LINsize) #applied after summing

        #two coords of size 4 and 4 addes
        layers_g2 = [ nn.Linear( 2*I_CNN + 4 + 4  + Q_GRU_out , LINsize),
               nn.ReLU(inplace=True),
               #nn.Dropout(0.5),
               nn.Linear( LINsize, LINsize),
               nn.ReLU(inplace=True),
               #nn.Dropout(0.5),
               nn.Linear(LINsize,LINsize),
               nn.ReLU(inplace=True) ]

        self.g2 = nn.Sequential(*layers_g2)


        self.f2_phi = nn.Linear(LINsize,LINsize) #applied after summing


        layers_f2 = [ nn.Linear(LINsize + LINsize,1024),
                      nn.ReLU(inplace=True),
                      #nn.Dropout(0.5),
                      nn.Linear(1024,self.Ncls) ]


        self.fgamma = nn.Sequential(*layers_f2)


        def make_coords(N):
            c = []
            W = 448
            box = W/N
            rangex = 4
            for y in range(rangex):
                for x in range(rangex):
                    grid=[x*box, y*box, (x+1)*box, (y+1)*box]
                    c.append(grid)
            arr = np.array(c)/448
            return torch.from_numpy(arr).float()

        coords = make_coords(4)
        self.pool_coords = coords.to("cuda")


    def forward(self,wholefeat,pooled,box_feats,q_feats,box_coords,index):

        q_rnn  = self.QRNN(q_feats)
        
        counts = []
        total = q_feats.size(0)
        for i in range(total):

            idx = index[i]


            wholefeat_idx = wholefeat[i,...]
            pooled_idx = pooled[i,...]
            box_feats_idx = box_feats[i,:idx,:]
            q_rnn_idx  =  q_rnn[i,:].unsqueeze(0)
            box_coords_idx = box_coords[i,:idx,:]
            N =  int(idx) # number of boxes
            # add coordinates
            box_feats_coords_idx = torch.cat([box_feats_idx, box_coords_idx],dim=-1)
            qst = q_rnn_idx.unsqueeze(1).expand(N,N,-1)
            b_i = box_feats_coords_idx.unsqueeze(1).expand(-1,N,-1)
            b_j = box_feats_coords_idx.unsqueeze(0).expand(N,-1,-1)

            #TODO addd question feats to the pooled and unpoolee

            #add more dropout in f1_phi and f2_phi

            #TODO; add IOU overlaps and dot procut of object embeddings
            #also maybe overlap statistics

            #print (b_j.size())
            # concatenate all together
            b_full = torch.cat([b_i,b_j,qst],-1)
            b_full = b_full.view(N*N,-1)
            #print ('bfull',b_full.size())

            g1_out = self.g1(b_full)
            g1_out_reduce = g1_out.sum(0).squeeze()

            g1_norelu = self.f1_phi(g1_out_reduce)
            RNO = F.relu(g1_norelu)
            Npooled = pooled.size(1)
            pooled_aa = box_feats_coords_idx.unsqueeze(1).expand(-1,Npooled,-1)

            pooled_idx_coords = torch.cat([pooled_idx,self.pool_coords],dim=-1)
            qst_add = q_rnn_idx.unsqueeze(1).expand(N,Npooled,-1)

            pooled_bb = pooled_idx_coords.unsqueeze(0).expand(N,-1,-1)
            box_bg = torch.cat([pooled_aa,pooled_bb,qst_add],dim=-1)
            box_bg = box_bg.view(N*Npooled,-1)


            g2_out = self.g2(box_bg)
            g2_out_reduce = g2_out.sum(0).squeeze()

            g2_norelu = self.f2_phi(g2_out_reduce)
            RNOB = F.relu(g2_norelu)


            RN_full = torch.cat([RNO,RNOB],dim=-1)

            count = self.fgamma(RN_full)
            counts.append(count.unsqueeze(0))
        return torch.cat(counts,0).squeeze(1)

