import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import datetime
import shutil
import json
import sys

EPS = 1e-7


def assert_eq(real, expected):
    assert real == expected,\
        '{} (true) vs {} (expected)'.format(real, expected)


def assert_array_eq(real, expected):
    assert (np.abs(real-expected) < EPS).all(), \
        '{} (true) vs {} (expected)'.format(real, expected)


def parsejson(jsonpath):   
    if not os.path.exists(jsonpath):
        print ("JSON path not found!!")
        return None        
    js = json.load(open(jsonpath))    
    return js

def filetostr(filepath):    
    "return str of file to read"
    if not os.path.exists(filepath):
        print ("JSON path not found!!")
        return
    with open(filepath,'r') as f:
        data = f.read()
    return data
    

def load_folder(folder, suffix):
    """return files with suffix , return str if one file only"""
    imgs = []
    for f in sorted(os.listdir(folder)):
        if f.endswith(suffix):
            imgs.append(os.path.join(folder, f))
    if len(imgs) == 1:
        return imgs[0]
    return imgs


def load_imageid(folder):
    images = load_folder(folder, 'jpg')
    img_ids = set()
    for img in images:
        img_id = int(img.split('/')[-1].split('.')[0].split('_')[-1])
        img_ids.add(img_id)
    return img_ids


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def weights_init(m):
    """custom weights initialization called on net_g and net_f."""
    cname = m.__class__
    if cname == nn.Linear or cname == nn.Conv2d or cname == nn.ConvTranspose2d:
        m.weight.data.normal_(0.0, 0.02)
    elif cname == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    else:
        print ('{} is not initialized.'.format(cname))


def init_net(net, net_file):
    if net_file:
        net.load_state_dict(torch.load(net_file))
    else:
        net.apply(weights_init)


def get_current_time():
    return datetime.datetime.now().strftime("%B %d, %Y - %I:%M%p")


def RMSE(true,pred):
    if isinstance(true,list):
        true = np.array(true,dtype=np.uint64)
    if isinstance(pred,list):
        pred = np.array(pred,dtype=np.uint64)
    diff = true - pred
    N = len(true)
    mse = np.sum(diff**2,axis=0)/ float(N)
    rmse = np.sqrt(mse)
    return rmse

def accuracy(true,pred):
    if isinstance(true,list):
        true = np.array(true,dtype=np.uint64)
    if isinstance(pred,list):
        pred = np.array(pred,dtype=np.uint64)
    N = len(true)
    return 100.0* np.sum(true == pred)/float(N)


# maintain all metrics required in this dictionary- these are used
#in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    'rmse': RMSE,
    # could add more metrics such as accuracy for each token type
}


def adjust_learning_rate(optimizer, newLR):
    """Sets the learning rate to newLR"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = newLR


def save_checkpoint(savefolder,tbs, is_best=False):
    epoch = tbs['epoch']
    
    if not os.path.exists(savefolder):
        os.mkdir(savefolder)
    
    filename = os.path.join(savefolder,'chkpoint_{}.pth'.format(epoch))
    bestfile = os.path.join(savefolder,'model_best.pth')
    torch.save(tbs, filename)
    if is_best:
        shutil.copyfile(filename, bestfile)

def load_checkpoint(filename,model,optimizer):
     #resume = 'a/checkpoint-1.pth'
     meta = {}  
     if os.path.exists(filename):
         checkpoint = torch.load(filename)
         epoch = checkpoint['epoch']
         print("=> loading checkpoint '{}' at epoch: {}".format(filename,epoch))
         model.load_state_dict(checkpoint['state_dict'])
         optimizer.load_state_dict(checkpoint['optimizer'])
         print("=> loaded checkpoint '{}' (epoch {})"
               .format(filename, epoch))
            
         for key in checkpoint:
             if key not in ['epoch','state_dict','optimizer']:
                 meta[key] = checkpoint[key]
         
         return epoch+1 , meta
     else:
         print("=> no checkpoint found at '{}'".format(filename))
         return 0, meta


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MovingMeanMonitor:
        """ Take an exponentially moving mean over the given values """
        name = 'mean'

        def __init__(self, momentum=0.9):
            self.momentum = momentum
            self.first = True
            self.value = None

        def update(self, value):
            if self.first:
                self.value = value
                self.first = False
            else:
                m = self.momentum
                self.value = m * self.value + (1 - m) * value



class Logger(object):
    def __init__(self, output_name):
        dirname = os.path.dirname(output_name)
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        else:
            feedback = input("Log folder already exists !!!! \nOverwrite & Continue [y/Y]?: ")
            if feedback not in ['y','Y']:
                print ("Restart and Save the logfile under different name.")
                sys.exit(0)

        self.dirname = dirname
        self.log_file = open(output_name, 'a+')
        self.infos = {}

    def append(self, key, val):
        vals = self.infos.setdefault(key, [])
        vals.append(val)


    def dump_info(self,filename="infos.json"):
        "dump infos in infos.json file"
        infopath = os.path.join(self.dirname,filename)
        with open(infopath,'w') as f:
            json.dump(self.infos,f)
        
    def log(self, extra_msg=''):
        msgs = [extra_msg]
        for key, vals in self.infos.iteritems():
            msgs.append('%s %.6f' % (key, np.mean(vals)))
        msg = '\n'.join(msgs)
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        self.infos = {}
        return msg

    def write_silent(self, msg):
        self.log_file.write(msg + '\n')
        self.log_file.flush()

    def write(self, msg):
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        print (msg)
