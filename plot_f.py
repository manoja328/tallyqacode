import argparse
import torch
import os
import config
from models.RN_GTU import RN
from utils import load_checkpoint
import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from models.language import getglove
from data import CountDataset

image_features_path = config.global_config['genome_bottomup'] 
features_file = h5py.File(image_features_path, 'r')

def _create_coco_id_to_index():
    """ Create a mapping from a COCO image id into the corresponding index into the h5 file """
    with h5py.File(image_features_path, 'r') as features_file:
        coco_ids = features_file['ids'][()]
    coco_id_to_index = {id: i for i, id in enumerate(coco_ids)}
    return coco_id_to_index   
     
def get_image_name_old(subtype='train2014', image_id='1', format='%s/COCO_%s_%012d.jpg'):
    return format%(subtype, subtype, image_id)

def retbox(x):
    return np.array([[x[0],x[0],x[2],x[2],x[0]],[x[1],x[3],x[3],x[1],x[1]]]).T
         

def load_image_coco(image_id):
    """ Load an image """      
    index = coco_id_to_index[image_id]
    L = features_file['num_boxes'][index]
    W = features_file['widths'][index]
    H = features_file['heights'][index]
    box_feats = features_file['features'][index]
    box_locations = features_file['boxes'][index]   
    return L,W,H,box_feats.T,box_locations.T

coco_id_to_index = _create_coco_id_to_index()    
cocoids =  list(coco_id_to_index.keys())

def saveimage(ent,vals):
    C = sum(vals)
    image = os.path.join('/home/manoj',ent['image'])
    image_id = ent['image_id']
    #for our dataset where ID has 9 in front 
    if 'VG_100K' in ent['image']:
        image_id = int(str(image_id)[1:]) #remove 9
    if image_id in coco_id_to_index:
        L,W,H,_,boxes =  load_image_coco(image_id)
        npimg = Image.open(image)      
#        print (npimg.width,npimg.height)
        plt.figure()
        plt.imshow(npimg)
        for i in range(L):
           xmin , ymin,xmax,ymax  = boxes[i]
           x =[xmin,ymin,xmax,ymax]
           rect = retbox(x)
           val = vals[i]
           plt.plot(rect[:,0],rect[:,1],'y',linewidth=1.0)
           plt.text(rect[0,0], rect[0,1],"{:.2f}".format(val),color='r', fontsize=10)
#        plt.axis('off')
        imglast = image.split("/")[-1]
        plt.title("Total Count: {:.2f} GT: {}".format(C,ent['answer']))
        plt.xlabel("{}".format(ent['question']))
        plt.savefig("ann_{}".format(imglast),dpi=150)

    else:
        print ("Image-id {} not found".format(image_id))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dsname', help='dataset: Ourdb | HowmanyQA' , default='HowmanyQA')
    parser.add_argument('--epochs', type=int,help='Number of epochs',default=50)
    parser.add_argument('--model', help='Model Q | I| QI | Main | RN',default='RN')
    parser.add_argument('--save', help='save folder name',default='NAC')
    parser.add_argument('--resume', type=str, default=None, help='resume file name')
    parser.add_argument('--q', type=str, default="How many dogs?", help='question')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = parse_args()
       
    isVQAeval = False
    if args.dsname == 'HowmanyQA':
        isVQAeval = True

    ds = config.dataset[args.dsname]
    N_classes = ds['N_classes']
    savefolder = '_'.join([args.dsname,args.model,args.save])


    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")    
    loader_kwargs = {'num_workers': 4} if use_cuda else {}

    model = RN(N_classes,debug = True)
    model = model.to(device)
    print (model)

    optimizer = torch.optim.Adam(model.parameters())
    start_epoch = 0
    if args.resume:
         start_epoch,meta = load_checkpoint(args.resume,model,optimizer)
         
    image_id = np.random.choice(cocoids)
    L, W, H ,box_feats,box_coords = load_image_coco(image_id)    

    testds = CountDataset(file = ds['test'],**config.global_config)
    testset = testds.data

    ent = np.random.choice(testset)
    print (ent)

    q_feats = getglove(ent['question'])
    q_feats = torch.from_numpy(q_feats)
    box_feats = torch.from_numpy(box_feats)

    box_feats = box_feats.to(device).unsqueeze(0)
    q_feats = q_feats.to(device).unsqueeze(0)

    net_kwargs = { 'wholefeat': None,
               'pooled' : None,
               'box_feats':box_feats,
               'q_feats':q_feats,
               'box_coords':None,
               'index':[L] }

    out,fvals = model(**net_kwargs)
    print ("Count val: ",out.item())
    fvals = fvals.squeeze(1).tolist()
    print (fvals)
    saveimage(ent,fvals)
         


