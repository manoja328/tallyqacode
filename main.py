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
import inspect

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dsname', help='dataset: Ourdb | HowmanyQA' , default='Ourdb')
    parser.add_argument('--epochs', type=int,help='Number of epochs',default=50)
    parser.add_argument('--model', help='Model Q | I| QI | Main | RN',default='Q')
    parser.add_argument('--lr', type=float,default=0.001,help='Learning rate')
    parser.add_argument('--bs', type=int,default=128,help='Batch size')
    parser.add_argument('--save', help='save folder name',default='0a1rand')
    parser.add_argument('--savefreq', help='save model frequency',type=int,default=1)
#    parser.add_argument('--isVQAeval', help='vqa eval or not',type=bool,default=True)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--expl', type=str, default='info', help='extra explanation of the method')
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
    logger.write("== {} ==".format(args.expl))

    

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")    
    kwargs = {'num_workers': 2, 'pin_memory': False} if use_cuda else {}


    testset = CountDataset(file = ds['test'],**config.global_config)
    trainset = CountDataset(file = ds['train'],istrain=True,**config.global_config)

    testloader = DataLoader(testset, batch_size=args.bs,
                             shuffle=False, **kwargs)
    trainloader = DataLoader(trainset, batch_size=args.bs,
                         shuffle=True, **kwargs)

    models = { 'Q':Qmodel, 'I': Imodel, 'QI': QImodel ,'RN': RN }
    
    model = models[args.model](N_classes)
    model = model.to(device)
    print (model)
    
    #log source code of model being used
    logger.write_silent(inspect.getsource(type(model)))
    logger.write_silent(repr(model))

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
