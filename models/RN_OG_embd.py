#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 13:04:34 2018

@author: manoj
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .lang_new import QuestionParser


class RN(nn.Module):
    def __init__(self,Ncls,**kwargs):
        super().__init__()

        I_CNN = 2048
        Q_GRU_out = 1024
        Q_embedding = 300
        LINsize = 1024
        Boxcoords = 16

        self.Ncls = Ncls
        #self.Ncls = 1

        if kwargs.get('trainembd'):       
            self.QRNN = QuestionParser(dictionaryfile = kwargs['dictionaryfile'],
                                       glove_file = kwargs['glove'],
                                         dropout=0.3, word_dim=Q_embedding,
                                         ques_dim=Q_GRU_out ,
                                         rnn_type= 'GRU')


        layers_g1 = [ nn.Linear( 2*I_CNN + Boxcoords + Q_GRU_out, LINsize),
               nn.ReLU(inplace=True),
               #nn.Dropout(0.5),
               nn.Linear( LINsize, LINsize),
               nn.ReLU(inplace=True),
               #nn.Dropout(0.5),
               nn.Linear(LINsize,LINsize),
               nn.ReLU(inplace=True)]

        self.g1 = nn.Sequential(*layers_g1)


        D = LINsize//2

        layers_f2 = [ nn.Linear(LINsize,D),
                      nn.ReLU(inplace=True),
                      #nn.Dropout(0.5),
                      nn.Linear(D,self.Ncls) ]


        self.fgamma = nn.Sequential(*layers_f2)


    @staticmethod
    def get_spatials(b):
        # b = (B, k, 6)

        b = b.float()

        k, _ = b.size()
        

        b_ij = b.unsqueeze(1).expand(-1,k,-1)  # (k, k, 6)
        b_ji = b.unsqueeze(0).expand(k,-1,-1)

        area_ij = (b_ij[..., 2] - b_ij[..., 0]) * (b_ij[..., 3] - b_ij[..., 1])
        area_ji = (b_ji[..., 2] - b_ji[..., 0]) * (b_ji[..., 3] - b_ji[..., 1])

        righmost_left = torch.max(b_ij[..., 0], b_ji[..., 0])
        downmost_top = torch.max(b_ij[..., 1], b_ji[..., 1])
        leftmost_right = torch.min(b_ij[..., 2], b_ji[..., 2])
        topmost_down = torch.min(b_ij[..., 3], b_ji[..., 3])

        # calucate the separations
        left_right = (leftmost_right - righmost_left)
        up_down = (topmost_down - downmost_top)

        # don't multiply negative separations,
        # might actually give a postive area that doesn't exit!
        left_right = torch.max(0*left_right, left_right)
        up_down = torch.max(0*up_down, up_down)

        overlap = left_right * up_down

        iou = overlap / (area_ij + area_ji - overlap)
        o_ij = overlap / area_ij
        o_ji = overlap / area_ji

        iou = iou.unsqueeze(-1)  # (k, k, 1)
        o_ij = o_ij.unsqueeze(-1)  # (k, k, 1)
        o_ji = o_ji.unsqueeze(-1)  # (k, k, 1)

        return b_ij, b_ji, iou, o_ij, o_ji

    def forward(self,wholefeat,pooled,box_feats,q_feats,box_coords,index):


        q_rnn  = self.QRNN(q_feats)

        B,_ = q_feats.size()

        counts = []
        total = q_feats.size(0)
        for i in range(total):

            idx = index[i]

            box_feats_idx = box_feats[i,:idx,:]
            q_rnn_idx  =  q_rnn[i,:].unsqueeze(0)
            box_coords_idx = box_coords[i,:idx,:]
            N =  int(idx) # number of boxes

            qst = q_rnn_idx.unsqueeze(1).expand(N,N,-1)
            
            o_i = box_feats_idx.unsqueeze(1).expand(-1,N,-1)
            o_j = box_feats_idx.unsqueeze(0).expand(N,-1,-1)
            
            # dot product: (B, k, k)
            vtv = torch.mul(o_i.contiguous().view(N*N,-1) , o_j.contiguous().view(N*N,-1))
            vtv = torch.sum(vtv,dim=1)
            dot = vtv.view(N,N,-1)
    
            b_ij, b_ji, iou, o_ij, o_ji = self.get_spatials(box_coords_idx)                        
            features = [ dot, b_ij , b_ji ,iou ,o_ij, o_ji]  # (k, k, 6)
            features = torch.cat(features, dim=-1)  # ( k, k, 17)

            boxes_full = torch.cat([o_i,o_j,qst,features],-1)
            boxes_full = boxes_full.view(N*N,-1)
            #print ('bfull',b_full.size())

            g1_out = self.g1(boxes_full)
            g1_out_reduce = g1_out.sum(0).squeeze()

            count = self.fgamma(g1_out_reduce)
            counts.append(count.unsqueeze(0))
        return torch.cat(counts,0).squeeze(1)


    