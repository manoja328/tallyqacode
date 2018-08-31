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
        Boxcoords = 0

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

        B, k, _ = b.size()

        b_ij = torch.stack([b] * k, dim=1)  # (B, k, k, 6)
        b_ji = b_ij.transpose(1, 2)

        area_ij = (b_ij[:, :, :, 2] - b_ij[:, :, :, 0]) * (b_ij[:, :, :, 3] - b_ij[:, :, :, 1])
        area_ji = (b_ji[:, :, :, 2] - b_ji[:, :, :, 0]) * (b_ji[:, :, :, 3] - b_ji[:, :, :, 1])

        righmost_left = torch.max(b_ij[:, :, :, 0], b_ji[:, :, :, 0])
        downmost_top = torch.max(b_ij[:, :, :, 1], b_ji[:, :, :, 1])
        leftmost_right = torch.min(b_ij[:, :, :, 2], b_ji[:, :, :, 2])
        topmost_down = torch.min(b_ij[:, :, :, 3], b_ji[:, :, :, 3])

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

        iou = iou.unsqueeze(3)  # (B, k, k, 1)
        o_ij = o_ij.unsqueeze(3)  # (B, k, k, 1)
        o_ji = o_ji.unsqueeze(3)  # (B, k, k, 1)

        return b_ij, b_ji, iou, o_ij, o_ji

    def forward(self,wholefeat,pooled,box_feats,q_feats,box_coords,index):


        q_rnn  = self.QRNN(q_feats)

        B = q_feats.size()

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

            features = []
#    
#            norm_v_emb = F.normalize(box_feats, dim=2)  # (B, k, v_dim)
#            vtv = torch.bmm(norm_v_emb, norm_v_emb.transpose(1, 2))  # (B, k, k)
#            vtv = vtv[:, :, :, None].repeat(1, 1, 1, 1)  # (B, k, k, 1)
#            assert vtv.size() == (B, N, N, 1)
#            features.append(vtv)
#    
#            b_ij, b_ji, iou, o_ij, o_ji = self.get_spatials(box_coords_idx)
#            
#            assert b_ij.size() == (B, N, N, 6)
#            assert b_ji.size() == (B, N ,N, 6)
#            assert iou.size()  ==  (B, N, N, 1)
#            assert o_ij.size() == (B, N, N, 1)
#            assert o_ji.size() == (B, N, N, 1)
#            
#            features.append(b_ij)  # (B, k, k, 6)
#            features.append(b_ji)  # (B, k, k, 6)
#            features.append(iou)  # (B, k, k, 1)
#            features.append(o_ij)  # (B, k, k, 1)
#            features.append(o_ji)  # (B, k, k, 1)
#            
#            features = torch.cat(features, dim=-1)  # (B, k, k, 17)

            boxes_full = torch.cat([o_i,o_j,qst],-1)
            boxes_full = boxes_full.view(N*N,-1)
            #print ('bfull',b_full.size())

            g1_out = self.g1(boxes_full)
            g1_out_reduce = g1_out.sum(0).squeeze()

            count = self.fgamma(g1_out_reduce)
            counts.append(count.unsqueeze(0))
        return torch.cat(counts,0).squeeze(1)


    