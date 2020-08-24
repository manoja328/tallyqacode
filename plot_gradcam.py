import argparse
import torch
import torch.nn as nn
import os
import config
from data import CountDataset
from torch.utils.data import  DataLoader
import torch.nn.functional as F
from utils import load_checkpoint
import sys
import numpy as np
from collections import OrderedDict
import h5py
import pickle
import matplotlib.pyplot as plt
from PIL import Image

DIR = 'gradcam_outputs'
if not os.path.exists(DIR):
    os.mkdir(DIR)


#%%
def retbox(x):
    return np.array([[x[0],x[0],x[2],x[2],x[0]],[x[1],x[3],x[3],x[1],x[1]]]).T
         
def saveimage(image,boxes,vals,pred):
    image = os.path.join('/home/manoj',ent['image'])
    npimg = Image.open(image)      
    plt.figure()
    plt.imshow(npimg)
    for i in range(L):
       xmin , ymin,xmax,ymax  = boxes[i]
       x =[xmin,ymin,xmax,ymax]
       rect = retbox(x)
       val = vals[i]
       #plt.plot(rect[:,0],rect[:,1],'r',linewidth=val)
       plt.plot(rect[:,0],rect[:,1],'r',alpha=val)
#       plt.text(rect[0,0], rect[0,1],"{:.2f}".format(val),color='r', fontsize=10)

    imglast = image.split("/")[-1]
    plt.title("Prediction: {} Ground truth: {}".format(pred,ent['answer']))
    plt.xlabel("{}".format(ent['question']))
    plt.savefig("{}/ann_{}.jpeg".format(DIR,ent['question_id']),dpi=150)
    plt.close()


#def saveimage_clean(ent,boxes):
#    image = os.path.join('/home/manoj',ent['image'])
#    imglast = image.split("/")[-1]
#    image_id = getimageid(ent)
#    if image_id in coco_id_to_index:
#        npimg = Image.open(image)      
#        plt.figure()
#        plt.imshow(npimg)
#        L = len(boxes)
#        for i in range(L):
#           xmin , ymin,xmax,ymax  = boxes[i]
#           x =[xmin,ymin,xmax,ymax]
#           rect = retbox(x)
#           plt.plot(rect[:,0],rect[:,1],'r',linewidth=1.0)
#        plt.savefig("rounding_test/annnms__{}".format(imglast),dpi=150)
#        plt.close()
#    else:
#        print ("Image-id {} not found".format(image_id))    


class BoxDataset():

    def __init__(self):
        
        kwargs = config.global_config
        
        with open(config.dataset['Ourdb']['test'],'rb') as f:
            self.data = pickle.load(f)
        
            
        self.pool_features_path_coco = kwargs.get('coco_pool_features')
        self.pool_features_path_genome = kwargs.get('genome_pool_features')

        self.image_features_path_coco = kwargs.get('coco_bottomup')
        self.coco_id_to_index =  self.id_to_index(self.image_features_path_coco)  
        self.image_features_path_genome = kwargs.get('genome_bottomup')
        self.genome_id_to_index =  self.id_to_index(self.image_features_path_genome)        

    def id_to_index(self,path):
        """ Create a mapping from a COCO image id into the corresponding index into the h5 file """
               
        with  h5py.File(path, 'r') as features_file:
            coco_ids = features_file['ids'][:]
        coco_id_to_index = {name: i for i, name in enumerate(coco_ids)}
        return coco_id_to_index       
        
    
    def _load_image_coco(self, image_id):
        """ Load an image """
        if not hasattr(self, 'features_file'):
            # Loading the h5 file has to be done here and not in __init__ because when the DataLoader
            # forks for multiple works, every child would use the same file object and fail
            # Having multiple readers using different file objects is fine though, so we just init in here.
            self.features_file = h5py.File(self.image_features_path_coco, 'r')
           
        index = self.coco_id_to_index[image_id]
        L = self.features_file['num_boxes'][index]
        W = self.features_file['widths'][index]
        H = self.features_file['heights'][index]
        box_feats = self.features_file['features'][index]
        box_locations = self.features_file['boxes'][index]
        return L,W,H,box_feats.T, box_locations.T 
 

    def _load_image_genome(self, image_id):
        """ Load an image """
        if not hasattr(self, 'features_file_genome'):
            # Loading the h5 file has to be done here and not in __init__ because when the DataLoader
            # forks for multiple works, every child would use the same file object and fail
            # Having multiple readers using different file objects is fine though, so we just init in here.
            self.features_file_genome = h5py.File(self.image_features_path_genome, 'r')
           
        image_id = int(str(image_id)[1:])
        index = self.genome_id_to_index[image_id]
        L = self.features_file_genome['num_boxes'][index]
        W = self.features_file_genome['widths'][index]
        H = self.features_file_genome['heights'][index]
        box_feats = self.features_file_genome['features'][index]
        box_locations = self.features_file_genome['boxes'][index]
        return L,W,H,box_feats.T, box_locations.T 
          

    def __getitem__(self, idx):
        ent = self.data[idx]       
        qid = ent['question_id']
        img_name = ent['image']
        img_id = ent['image_id']
        if 'VG' in img_name:
            L, W, H ,imgarr,box_coords = self._load_image_genome(img_id)
        else:
            L, W, H ,imgarr,box_coords = self._load_image_coco(img_id)

        return W,H,box_coords,L,ent

#%%
        
    
class _PropagationBase(object):
    def __init__(self, model):
        super(_PropagationBase, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model

    def _encode_one_hot(self, idx):
        one_hot = torch.FloatTensor(1, self.preds.size()[-1]).zero_()
        one_hot[0][idx] = 1.0
        return one_hot.to(self.device)

    def forward(self, **kwargs):      
        self.preds = self.model(**kwargs)
        print ("out ",self.preds)
        self.prob = F.softmax(self.preds,dim=-1)
        _,clspred = torch.max(self.prob,-1)
        return self.prob, clspred

    def backward(self, idx):
        one_hot = self._encode_one_hot(idx)
        self.preds.backward(gradient=one_hot, retain_graph=True)

#%%

class GradCAM(_PropagationBase):
    def __init__(self, model):
        super(GradCAM, self).__init__(model)
        self.all_fmaps = OrderedDict()
        self.all_grads = OrderedDict()

        def func_f(module, input, output):
            self.all_fmaps[id(module)] = output[0].detach()

        def func_b(module, grad_in, grad_out):
            if grad_out[0] is None:
                return
            self.all_grads[id(module)] = grad_out[0].detach()

        for module in self.model.named_modules():
            module[1].register_forward_hook(func_f)
            module[1].register_backward_hook(func_b)

    def _find(self, outputs, target_layer):
        for key, value in outputs.items():
            for module in self.model.named_modules():
                if id(module[1]) == key:
                    if module[0] == target_layer:
                        return value
        raise ValueError('Invalid layer name: {}'.format(target_layer))

    def _normalize(self, grads):
        l2_norm = torch.sqrt(torch.mean(torch.pow(grads, 2))) + 1e-5
        return grads / l2_norm
    
    def generate(self, target_layer):   
        fmaps = self._find(self.all_fmaps,target_layer)
        grads = self._find(self.all_grads,target_layer)
        print ("fmaps: {} weights {}".format(fmaps.shape,grads.shape))
        grads = self._normalize(grads)   
        
        if 'g2' in target_layer:
            Nboxes = len(grads) / 16
        
        else :
            Nboxes = len(grads)** 0.5
            
        Nboxes = int(Nboxes)    
        s = torch.sum(fmaps * grads,dim=1)
        s = torch.clamp(s,min=0)
        s = (s - s.min())/ ( s.max() - s.min())
        #gs = s.view(index,-1).mean(dim=1)
        gs,_ = s.view(Nboxes,-1).max(dim=1)
        gs = (gs - gs.min())/ ( gs.max() - gs.min())    
        return gs
    
    


qidtods = {}
for i,ent in enumerate(pickle.load(open(config.dataset['Ourdb']['test'],'rb'))):
    qidtods[ent['question_id']] = i 
    
    
boxdataset = BoxDataset()    
#%%

def instance_bce_with_logits(logits, labels):
    assert logits.dim() == 2
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss

#%%

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dsname', help='dataset: Ourdb | HowmanyQA' , default='Ourdb')
    parser.add_argument('--epochs', type=int,help='Number of epochs',default=50)
    parser.add_argument('--model', help='Model Q | I| QI | Main | RN',default='RN_BGOG_embdirlc')
    parser.add_argument('--lr', type=float,default=0.0003,help='Learning rate')
    parser.add_argument('--bs', type=int,default=32,help='Batch size')
    parser.add_argument('--save', help='save folder name',default="sigmoid2")
    parser.add_argument('--savefreq', help='save model frequency',type=int,default=1)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--resume', type=str, default="Ourdb_RN_BGOG_embdirlc_sigmoid2/chkpoint_8.pth", help='resume file name')
    parser.add_argument('--test', type=bool, default=False, help='test only')
    parser.add_argument('--testrun', action='store_true', help='test run with few dataset')
    parser.add_argument('--isnms', type=bool, default=False, help='Do nms?')
    parser.add_argument('--trainembd',type=bool,default=True,help='use fixed / trainable embedding')
    parser.add_argument('--nobaselines', action='store_true',help='does not eval baselines')
    parser.add_argument('--savejson',type=bool,default=True,help='save json in VQA format')
    parser.add_argument('--clip_norm', type=float, default=200.0, help='norm clipping')
    parser.add_argument('--expl', type=str, default='info', help='extra explanation of the method')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = parse_args()
       
    isVQAeval = False
    if args.dsname != 'Ourdb':
        isVQAeval = True

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    ds = config.dataset[args.dsname]
    N_classes = ds['N_classes']

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")    
    loader_kwargs = {'num_workers': 1} if use_cuda else {}
    model = config.models.get(args.model,None)
    if model is None:
        print ("Model name not found valid names are: {} ".format(config.models))
        sys.exit(0)
    model = model(N_classes,trainembd=args.trainembd,**config.global_config)
    model = model.to(device)
    
    savefolder = '_'.join([args.dsname,args.model,args.save])
    print (model)

    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
   
    start_epoch = 0
    if args.resume:
         start_epoch,meta = load_checkpoint(args.resume,model,optimizer)


    dskwargs = { 'trainembd':args.trainembd , 'isnms':args.isnms ,
                'testrun':args.testrun , **config.global_config}
    testds = CountDataset(file = ds['test'],istrain=False,**dskwargs)   

    test_loader = DataLoader(testds, batch_size= 1,
                             shuffle=False, **loader_kwargs)

    
    clslossfn = instance_bce_with_logits
    
    kwargs = {  **args.__dict__,
                    'start_epoch': start_epoch,
                     'jsonfolder': config.global_config['jsonfolder'],
                     'N_classes': N_classes,
                     'savefolder': savefolder, 
                     'isVQAeval': isVQAeval,
                     'device' : device, 
                     'model' :  model,
                     'test_loader': test_loader,
                     'optimizer' : optimizer,                 
                  }
        
         
    it = iter(test_loader)
    
    device = kwargs.get('device')                    
    optimizer = kwargs.get('optimizer')
    istrain = kwargs.get('istrain')    
    gcam = GradCAM(model=model)
    
#%%
    for data in test_loader:
        qid,wholefeat,pooled,boxes,labels,targets,ques,box_coords,index = data  
        
        print ("Qid: {} GT: {}".format(qid,labels))
        
        labels = labels.long()            
        index  = index.long()
        B = qid.size(0)
        #converts 14_14 to 7_7
        #change pool size
        
        if torch.sum(pooled):
            pooled = F.avg_pool2d(pooled.permute(0,3,1,2),8,2)
            Npool = pooled.size(-1)
            pooled = pooled.view(B,2048,Npool**2)
            pooled = pooled.permute(0,2,1)
            pooled = F.normalize(pooled,p=2,dim=-1)
            #print (pooled.shape)
    
            pooled = pooled.to(device)
            wholefeat = F.normalize(wholefeat,p=2,dim=-1)
        else:
            pooled = wholefeat = None    
    
        
        
        boxes = F.normalize(boxes,p=2,dim=-1)
        box_feats = boxes.to(device)
        box_coords = box_coords.to(device)
        labels = labels.to(device)
        targets = targets.to(device)
        q_feats = ques.to(device)
                           
        net_kwargs = { 'wholefeat':wholefeat,
                       'pooled' :pooled,
                       'box_feats':box_feats,
                       'q_feats':q_feats,
                       'box_coords':box_coords,
                       'index':index}
    
        
        optimizer.zero_grad()
        probs, idx = gcam.forward(**net_kwargs)
        gcam.backward(idx= idx)
        gs = gcam.generate(target_layer= 'g2.0')       
        idx_ds = qidtods[qid.item()]
        W,H,box_coords,L,ent  = boxdataset.__getitem__(idx_ds)
        print (ent)
        saveimage(ent,box_coords,gs,idx.item())