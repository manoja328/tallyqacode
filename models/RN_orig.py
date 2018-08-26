#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 15:22:53 2018

@author: manoj

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationalLayerBase(nn.Module):
    def __init__(self, in_size, out_size, qst_size, hyp):
        super().__init__()

        self.f_fc1 = nn.Linear(hyp["g_layers"][-1], hyp["f_fc1"])
        self.f_fc2 = nn.Linear(hyp["f_fc1"], hyp["f_fc2"])
        self.f_fc3 = nn.Linear(hyp["f_fc2"], out_size)    
        self.dropout = nn.Dropout(p=hyp["dropout"])
        self.hyp = hyp
        self.qst_size = qst_size
        self.in_size = in_size
        self.out_size = out_size


        
class RelationalLayer(RelationalLayerBase):
    def __init__(self, in_size, out_size, qst_size, hyp, extraction=False):
        super().__init__(in_size, out_size, qst_size, hyp)

        self.quest_inject_position = hyp["question_injection_position"]
        self.in_size = in_size

	    #create all g layers
        self.g_layers = []
        self.g_layers_size = hyp["g_layers"]
        for idx,g_layer_size in enumerate(hyp["g_layers"]):
            in_s = in_size if idx==0 else hyp["g_layers"][idx-1]
            out_s = g_layer_size
            if idx==self.quest_inject_position:
                #create the h layer. Now, for better code organization, it is part of the g layers pool. 
                l = nn.Linear(in_s+qst_size, out_s)
            else:
                #create a standard g layer.
                l = nn.Linear(in_s, out_s)
            self.g_layers.append(l)	
        self.g_layers = nn.ModuleList(self.g_layers)
        self.extraction = extraction
    
    def forward(self, x, qst):
        # x = (B x 8*8 x 24)
        # qst = (B x 128)
        """g"""
        b, d, k = x.size()
        qst_size = qst.size()[1]
        
        # add question everywhere
        qst = torch.unsqueeze(qst, 1)                      # (B x 1 x 128)
        qst = qst.repeat(1, d, 1)                       # (B x 64 x 128)
        qst = torch.unsqueeze(qst, 2)                      # (B x 64 x 1 x 128)
        
        # cast all pairs against each other
        x_i = torch.unsqueeze(x, 1)                   # (B x 1 x 64 x 26)
        x_i = x_i.repeat(1, d, 1, 1)                    # (B x 64 x 64 x 26)
        x_j = torch.unsqueeze(x, 2)                   # (B x 64 x 1 x 26)
        #x_j = torch.cat([x_j, qst], 3)
        x_j = x_j.repeat(1, 1, d, 1)                    # (B x 64 x 64 x 26)
        
        # concatenate all together
        x_full = torch.cat([x_i, x_j], 3)                  # (B x 64 x 64 x 2*26)
        
        # reshape for passing through network
        x_ = x_full.view(b * d**2, self.in_size)

        #create g and inject the question at the position pointed by quest_inject_position.
        for idx, (g_layer, g_layer_size) in enumerate(zip(self.g_layers, self.g_layers_size)):
            if idx==self.quest_inject_position:
                in_size = self.in_size if idx==0 else self.g_layers_size[idx-1]

                # questions inserted
                x_img = x_.view(b,d,d,in_size)
                qst = qst.repeat(1,1,d,1)
                x_concat = torch.cat([x_img,qst],3) #(B x 64 x 64 x 128+256)

                # h layer
                x_ = x_concat.view(b*(d**2),in_size+self.qst_size)
                x_ = g_layer(x_)
                x_ = F.relu(x_)
            else:
                x_ = g_layer(x_)
                x_ = F.relu(x_)

        if self.extraction:
            return None
        
        # reshape again and sum
        x_g = x_.view(b, d**2, self.g_layers_size[-1])
        x_g = x_g.sum(1).squeeze(1)
        
        """f"""
        x_f = self.f_fc1(x_g)
        x_f = F.relu(x_f)
        x_f = self.f_fc2(x_f)
        x_f = self.dropout(x_f)
        x_f = F.relu(x_f)
        x_f = self.f_fc3(x_f)

        return F.log_softmax(x_f, dim=1)
    
    
    
    
    
class RN(nn.Module):
    def __init__(self,hyp, extraction=False):
        super(RN, self).__init__()
        self.coord_tensor = None
        self.on_gpu = False
        
                  
        # LSTM
        hidden_size = hyp["lstm_hidden"]
        self.text = nn.LSTM(hyp["lstm_word_emb"],hidden_size,num_layers=1,bidirectional=False)
        
        # RELATIONAL LAYER
        self.rl_in_size = hyp["rl_in_size"]
        self.rl_out_size =  hyp["rl_out_size"]
        self.rl = RelationalLayer(self.rl_in_size, self.rl_out_size, hidden_size, hyp, extraction) 
        if hyp["question_injection_position"] != 0:          
            print('Supposing IR model')
        else:     
            print('Supposing original DeepMind model')

    def forward(self, img, qst_idxs):
        if self.state_desc:
            x = img # (B x 12 x 8)
        else:
            x = self.conv(img)  # (B x 24 x 8 x 8)
            b, k, d, _ = x.size()
            x = x.view(b,k,d*d) # (B x 24 x 8*8)
            
            # add coordinates
            if self.coord_tensor is None or torch.cuda.device_count() == 1:
                self.build_coord_tensor(b, d)                  # (B x 2 x 8 x 8)
                self.coord_tensor = self.coord_tensor.view(b,2,d*d) # (B x 2 x 8*8)
            
            x = torch.cat([x, self.coord_tensor], 1)    # (B x 24+2 x 8*8)
            x = x.permute(0, 2, 1)    # (B x 64 x 24+2)
        
        qst = self.text(qst_idxs)
        y = self.rl(x, qst)
        return y
       
    # prepare coord tensor
    def build_coord_tensor(self, b, d):
        coords = torch.linspace(-d/2., d/2., d)
        x = coords.unsqueeze(0).repeat(d, 1)
        y = coords.unsqueeze(1).repeat(1, d)
        ct = torch.stack((x,y))
        # broadcast to all batches
        # TODO: upgrade pytorch and use broadcasting
        ct = ct.unsqueeze(0).repeat(b, 1, 1, 1)
        self.coord_tensor = torch.tensor(ct)
        if self.on_gpu:
            self.coord_tensor = self.coord_tensor.cuda()
    
            
hyp = {'dropout': 0.05,
 'f_fc1': 512,
 'f_fc2': 1024,
 'g_layers': [511, 512],
 'lstm_hidden': 256,
 'lstm_word_emb': 300,
 'question_injection_position': 1,
 'rl_in_size': 14,
 'rl_out_size': 1,
 'state_description': True}    
    
rn = RN(hyp=hyp)
    
    