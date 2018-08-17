import torch
import torch.nn as nn
import time
from utils import AverageMeter
from models.callbacks import EarlyStopping
import torch.nn.functional as F
from utils import save_checkpoint
import utils
import numpy as np
import eval_extra
#%%


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': utils.accuracy,
    'rmse': utils.RMSE,
    # could add more metrics such as accuracy for each token type
}


def main(**kwargs):
    
    device = kwargs.get('device') 
    net = kwargs.get('model')                     
    optimizer = kwargs.get('optimizer')
    epoch = kwargs.get('epoch')
    istrain = kwargs.get('istrain')
  
    start_time = time.time()
    true = []
    pred_reg = []
    pred_cls = []
    idxs = []
    loss_meter = AverageMeter()
    loss_meter.reset()
    
    Nprint = 100
    if istrain:
        net.train()
        loader = kwargs.get('train_loader')
    else:
        loader = kwargs.get('test_loader')
        net.eval()

    reglossfn = nn.SmoothL1Loss() # also known as huber loss
    #reglossfn = nn.MSELoss()
    #reglossfn = nn.L1Loss()
    clslossfn = nn.CrossEntropyLoss()

    for i, data in enumerate(loader):


        qid,wholefeat,pooled,boxes,labels,ques,box_coords,index = data
        
        idxs.extend(qid.tolist())
    
        labels = labels.long()
        
        index  = index.long()
        B = wholefeat.size(0)
        #converts 14_14 to 7_7
        #change pool size
        pooled = F.avg_pool2d(pooled.permute(0,3,1,2),8,2)
        Npool = pooled.size(-1)
        pooled = pooled.view(B,2048,Npool**2)
        pooled = pooled.permute(0,2,1)
        pooled = F.normalize(pooled,p=2,dim=-1)
        #print (pooled.shape)


        wholefeat = F.normalize(wholefeat,p=2,dim=-1)

        #box_coords_add1 = torch.cat([torch.ones(B,N,1),box_coords],dim=-1)
        #box_coords_add1 = F.normalize(box_coords_add1,dim=-1)

        true.extend(labels.tolist())

        #normalize the box feats
        boxes = F.normalize(boxes,p=2,dim=-1)
        box_feats = boxes.to(device)
        box_coords = box_coords.to(device)
        labels = labels.to(device)
        q_feats = ques.to(device)
#       coord_feats = Variable(coords.type(dtype))


        optimizer.zero_grad()
        
        
        net_kwargs = { 'wholefeat':wholefeat,
                       'pooled' :pooled,
                       'box_feats':box_feats,
                       'q_feats':q_feats,
                       'box_coords':box_coords,
                       'index':index}
        
               
        if istrain:
             out = net(**net_kwargs)
                
        else:
            with torch.no_grad():
                out = net(**net_kwargs)

#        #sometimes in a batch only 1 example at the end
#        if out.dim() == 1: # add one more dimension
#            out = out.unsqueeze(0)

#        #print("Dataloader : {:2.2f} s".format(time.time() - testtime))
#        if out.size(1) > 1: # if classification
#            #print ('using classification')
#            loss = clslossfn(out, labels.long())
#            _,clspred = torch.max(out,-1)
#            pred_reg.extend(clspred.data.cpu().numpy().ravel())

#        elif out.size(1) == 1:  # if regression
            #print ('using regression')
        loss = reglossfn(out,labels.float())
        #round the output
        regpred = torch.round(out.data.cpu()).numpy().ravel()
        pred_reg.extend(regpred)

        loss_meter.update(loss.item())

        if istrain:
            #scheduler.step()
            #optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()

        if i == 0 and epoch == 0 and istrain:
            print ("Starting loss: {:.4f}".format(loss.item()))


        if i % Nprint == Nprint-1:
            infostr = "Epoch [{}]:Iter [{}]/[{}] Loss: {:.4f} Time: {:2.2f} s"
            printinfo = infostr.format(epoch , i, len(loader),
                                       loss_meter.avg,time.time() - start_time)

            print (printinfo)

    print("Completed in: {:2.2f} s".format(time.time() - start_time))
    ent = {}
    ent['true'] = true
    ent['pred_reg'] = pred_reg
    ent['pred_cls'] = pred_reg
    ent['loss'] = loss_meter.avg
    ent['qids'] = idxs
    return ent



def run(**kwargs):

    #DETECT, MUTAN , Zhang , UPdown baselines
    eval_extra.main(**kwargs)  

    savefolder = kwargs.get('savefolder')
    logger = kwargs.get('logger')
    epochs = kwargs.get('epochs')
    isVQAeval = kwargs.get('isVQAeval')
    N_classes = kwargs.get('N_classes')
    test_loader = kwargs.get('test_loader')
    testset = test_loader.dataset.data
    early_stop = EarlyStopping(monitor='test_loss',patience=3)
    
    Modelsavefreq = 1

    for epoch in range(epochs):

        kwargs['epoch'] = epoch
        train = main(istrain=True,**kwargs)
        test =  main(istrain=False,**kwargs)


        logger.write('Epoch {}: ------'.format(epoch))
        logger.write('\tTrain Loss: {:.4f}'.format(train['loss']))
        logger.write('\tTest Loss: {:.4f}'.format(test['loss']))
        
        #clamp all output
        pred_reg = np.array(test['pred_reg'],dtype=np.uint64)
        pred_reg_clip = pred_reg.clip(min=0,max=N_classes-1).tolist()
        predictions = dict(zip(test['qids'] , pred_reg_clip))
        
        #convert to string for vqa eval
        
        if isVQAeval:
            acc,rmse = eval_extra.evalvqa(testset,predictions,isVQAeval)
            logger.write("\tRMSE:{:.2f} Accuracy {:.2f}%".format(rmse,acc))
          
        else:            
            simp_comp = eval_extra.eval_simp_comp(testset,predictions)
            for d in ['simple','complex']:
                acc,rmse = simp_comp[d]
                logger.write("\t{} RMSE:{:.2f} Accuracy {:.2f}%".format(d,rmse,acc))
            
                     
#        for metric in metrics:
#            metric.eval(train['true'],train['pred_reg'])

        is_best = False
        if epoch % Modelsavefreq == 0:
            print ('Saving model ....')
            tbs = {
                'epoch': epoch,
                'state_dict': kwargs.get('model').state_dict(),
                'true':test['true'],
                'pred_reg':test['pred_reg'],
                'pred_cls':test['pred_cls'],
                'optimizer' : kwargs.get('optimizer').state_dict(),
            }

            save_checkpoint(savefolder,tbs,is_best)

        early_stop.on_epoch_end(epoch,logs=test)
        if early_stop.stop_training:
            kwargs.get('optimizer').param_groups[0]['lr'] *= 0.8
            lr =  kwargs.get('optimizer').param_groups[0]['lr']
            logger.write("New Learning rate: ",lr)
            early_stop.reset()
            #break
    logger.write('Finished Training')


