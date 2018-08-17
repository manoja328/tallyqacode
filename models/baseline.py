import torch
import torch.nn as nn
from torchvision import models

class VGG16(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(pretrained=True)
        self.top = nn.Sequential(*list(vgg.children())[0])
        for p in self.top.parameters():
            p.requires_grad = False
        _bottom  = [*list(vgg.children())[1]]
        self.onlylast = nn.Sequential(*_bottom[:2])
        for p in self.onlylast.parameters():
            p.requires_grad = False

    def forward(self,x):
        x = self.top(x)
        x = x.view(-1,25088)
        x = self.onlylast(x)
        return x


def build_cnn():
    resnet152 = models.resnet152(pretrained=True)
    modules=list(resnet152.children())[:-1]
    new_classifier =nn.Sequential(*modules)
    for p in new_classifier.parameters():
        p.requires_grad = False
    return new_classifier


def build_mlp(input_dim, hidden_dims, output_dim,
              use_batchnorm=False, dropout=0):
  layers = []
  D = input_dim
  if dropout > 0:
      layers.append(nn.Dropout(p=dropout))
  if use_batchnorm:
      layers.append(nn.BatchNorm1d(input_dim))
  for dim in hidden_dims:
        layers.append(nn.Linear(D, dim))
        if use_batchnorm:
          layers.append(nn.BatchNorm1d(dim))
        if dropout > 0:
          layers.append(nn.Dropout(p=dropout))
        layers.append(nn.ReLU(inplace=True)) # can use leaky relu tooo
        D = dim
  layers.append(nn.Linear(D, output_dim))
  return nn.Sequential(*layers)



class MLBAtt(nn.Module):
  def __init__(self,dim_q=512,dim_h=1024):
      super().__init__()
      self.linear_v_fusion = nn.Linear(dim_q,dim_h)
  def forward(self, x_v, x_q):
      x_v = self.linear_v_fusion(x_v)
      x_att = torch.mul(x_v, x_q)
      return x_att


class QImodel(nn.Module):
    def __init__(self,Ncls):
        super().__init__()

        I_CNN = 2048
        Box_GRU_out = 1024
        Q_GRU_out = 512
        Q_embedding = 300
        att_out = 512

        self.BoxRNN = nn.LSTM(att_out,Box_GRU_out,num_layers=1,bidirectional=False)
        self.clshead = build_mlp( I_CNN + Q_GRU_out , [1024],1)      
        self.QRNN = nn.LSTM(Q_embedding,Q_GRU_out,num_layers=1,bidirectional=False)


    def forward(self,**kwargs):
        
        box_feats = kwargs.get("box_feats")
        q_feats = kwargs.get("q_feats")
                
        enc2,_ = self.QRNN(q_feats.permute(1,0,2))
        q_rnn = enc2[-1]
                
        I = box_feats[:,0,:]        
        qi = torch.cat([I,q_rnn],-1)
               
        cls_scores = self.clshead(qi)
        return cls_scores


class Qmodel(nn.Module):
    def __init__(self,Ncls):
        super().__init__()       
        Q_GRU_out = 512
        Q_embedding = 300        
        self.clshead = build_mlp( Q_GRU_out,[1024] , 1)     
        self.QRNN = nn.LSTM(Q_embedding,Q_GRU_out,num_layers=1,bidirectional=False)

    def forward(self,**kwargs): 
        q_feats = kwargs.get("q_feats")
        enc2,_ = self.QRNN(q_feats.permute(1,0,2))
        q_rnn = enc2[-1]
        cls_scores = self.clshead(q_rnn)
        return cls_scores.view(-1)

class Imodel(nn.Module):
    def __init__(self,Ncls):
        super().__init__()
        I_CNN = 2048
        self.clshead = build_mlp( I_CNN , [1024], 1)

    def forward(self,**kwargs):       
        box_feats = kwargs.get("box_feats")      
        I = box_feats[:,0,:]
        cls_scores = self.clshead(I)
        return cls_scores

