import argparse
import torch
import os
from utils import Logger
import config
from data import CountDataset
from torch.utils.data import  DataLoader
from train import run
from models.baseline import Qmodel,Imodel,QImodel
from models.RN_BGOGnew import RN

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dsname', help='dataset: Ourdb | HowmanyQA' , default='HowmanyQA')
    parser.add_argument('--epochs', type=int,help='Number of epochs',default=50)
    parser.add_argument('--model', help='Model Q | I| QI | Main | RN',default='Q')
    parser.add_argument('--lr', type=float,default=0.001,help='Learning rate')
    parser.add_argument('--bs', type=int,default=32,help='Batch size')
    parser.add_argument('--save', help='save folder name',default='01')
    parser.add_argument('--savefreq', help='save model frequency',type=int,default=1)
#    parser.add_argument('--isVQAeval', help='vqa eval or not',type=bool,default=True)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = parse_args()
       
    isVQAeval = False
    if args.dsname == 'HowmanyQA':
        isVQAeval = True

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    ds = config.dataset[args.dsname]
    N_classes = ds['N_classes']
    savefolder = '_'.join([args.dsname,args.model,args.save])
    logger = Logger(os.path.join(savefolder, 'log.txt'))
    

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")    
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    testset = CountDataset(file = ds['test'],**config.global_config)
    trainset = CountDataset(file = ds['train'],train=True,**config.global_config)

    testloader = DataLoader(testset, batch_size=args.bs,
                             shuffle=False, **kwargs)
    trainloader = DataLoader(trainset, batch_size=args.bs,
                         shuffle=True, **kwargs)

    models = { 'Q':Qmodel, 'I': Imodel, 'QI': QImodel ,'RN': RN }
    
    model = models[args.model](N_classes)
    model = model.to(device)
    print (model)

    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)

    run_kwargs = {   'jsonfolder': config.global_config['jsonfolder'],
                     'N_classes': N_classes,
                     'dsname': args.dsname,
                     'savefolder': savefolder, 
                     'isVQAeval': isVQAeval,
                     'device' : device, 
                     'model' :  model,
                     'train_loader': trainloader,
                     'test_loader': testloader,
                     'optimizer' : optimizer,
                     'epochs': args.epochs,
                     'logger':logger
                  }

    run(**run_kwargs)
