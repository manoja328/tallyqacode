#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 14:19:33 2018

@author: manoj
"""
from eval_extra import get_acc_rmse

def isRL(q):
    qs = q.split()    
    for rl in ['left','right','top','down','on','up','near']:
        if rl in qs:
            return True
    return False

def eval_simp_comp_RLstyle(evalset,predictions):
    
        simple_gt ,simple_pred , complex_gt , complex_pred = [],[],[],[]
        simp_comp = {}

        for ent in evalset:
            q  = ent['question']
            qid = ent['question_id']           
            gt,pred = ent['answer'] , predictions[qid]
                        
            if ent['data_source'] !='amt' and (ent['issimple'] == False):
                    continue
            if isRL(q):
                complex_gt.append(gt)
                complex_pred.append(pred)
                
            else:
                simple_gt.append(gt)
                simple_pred.append(pred)
                
        print ("RL: {} not_RL: {}".format(len(complex_gt),len(simple_gt)))

        Sacc, Srmse  =  get_acc_rmse(simple_gt,simple_pred) 
        Cacc , Crmse  =  get_acc_rmse(complex_gt,complex_pred)
        
        simp_comp['not_RL'] = (Sacc,Srmse)
        simp_comp['RL'] = (Cacc,Crmse)
        return simp_comp