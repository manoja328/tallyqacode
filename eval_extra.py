import pickle
import os
from utils import  accuracy,RMSE
from collections import Counter
from utils import assert_eq , parsejson,load_folder

def evalvqa(evalset,predictions,isVQAeval=True):
    predans = []
    MCans = []
    VQA_acc = 0
    
    assert_eq(len(evalset) , len(predictions))
    
    for i,ent in enumerate(evalset):
        #all qids are integers
        qid =  int(ent['question_id'])
        #all predictions/answers are string
        pred   =  str(predictions[qid])
        if isVQAeval:
            ansfreq = Counter([ ans['answer'] for ans in ent['answers']])
            MC = ent['multiple_choice_answer']
            #prediction is a string
            agreeing = ansfreq[pred]
            ans_p = min(agreeing * 0.3, 1)
        else: # not VQA style        
            MC = ent['answer']           
            ans_p = (MC == pred)
            
        VQA_acc += ans_p
        predans.append(int(pred))
        MCans.append(int(MC))

    VQA_acc = 100.0 * VQA_acc / len(evalset)
    rmse = RMSE(MCans,predans)
    return VQA_acc, rmse


def evalHQA(key,evalset,jsonpath=None):

    js = parsejson(jsonpath=jsonpath)    
    predictions = {}
    #HQA qids are string so convert them to int
    for qid,ent in js.items():
        qid = int(qid)
        if str.isdigit(key):
            pred  = int(key)
        else:
            pred  = int(ent[key])
               
        if pred>20: # clamp every regression values to 20
                pred = 20
        predictions[qid] = str(pred)
    
    result = evalvqa(evalset,predictions,isVQAeval=True)
    return result

    
def eval_zhang_updown_mutan(evalset,jsonpath=None):
    js = parsejson(jsonpath=jsonpath) 
    predictions = {}
    for ent in js:
        qid = ent['question_id']
        answer = ent['answer']
        if str.isdigit(answer):
            answer = str(answer)
        else: # sometimes 'many' is also an answer
            answer = '0'
            
        predictions[qid] = answer
        
    return predictions



def get_detect(evalset):
    predictions = {}
    detect_gt = []
    for ent in evalset:
        qid = ent['question_id']
        nouns =  ent['noun']
        ans = ent.get('multiple_choice_answer',None)
        if ans is None:
            ans = ent['answer']
        detect_gt.append(int(ans))
        img_name = ent['image']

        #js_arr = getjson(img_name,jsondir=ds['test']['jsondir'])
        lasttwo = '/'.join(img_name.split("/")[-2:])
        lasttwo +=".pkl"

        if not os.path.exists(os.path.join("feats",lasttwo)):
            print ("file not found", lasttwo)
        pk = pickle.load(open(os.path.join("feats",lasttwo),"rb"))
        count = 0
        N = 15
        for i,ent  in enumerate(pk[:-1]):
            if i == N:
                break
            cat_name = ent['noun']
            if nouns == []:
                continue
            if cat_name in nouns:
                count = count + 1
        predictions[qid] = count
    
    return detect_gt,predictions

    
    

def get_acc_rmse(gt,pred):    
    acc = accuracy(gt,pred)
    rmse = RMSE(gt,pred)
    return acc,rmse


def eval_guess(gt):
    pred = {}
    N = len(gt)
    for guess in [0,1,2]:
        pred_guess = [guess] *N
        acc , rmse = get_acc_rmse(gt,pred_guess)
        pred[guess] = (acc,rmse)
    return pred

#%%


def eval_simp_comp(evalset,predictions,baselines=False):
    
        simple_gt ,simple_pred , complex_gt , complex_pred = [],[],[],[]
        simp_comp = {}

        for ent in evalset:
            qid = ent['question_id']           
            gt,pred = ent['answer'] , predictions[qid]
            
            if ent['data_source'] == 'amt':
               complex_gt.append(gt)
               complex_pred.append(pred)
                
            else: # not in AMT and  simple
                if ent['issimple']:                   
                    simple_gt.append(gt)
                    simple_pred.append(pred)

        Sacc, Srmse  =  get_acc_rmse(simple_gt,simple_pred) 
        Cacc , Crmse  =  get_acc_rmse(complex_gt,complex_pred)
        
        simp_comp['simple'] = (Sacc,Srmse)
        simp_comp['complex'] = (Cacc,Crmse)

        if baselines:
            simple_guess = eval_guess(simple_gt)
            complex_guess = eval_guess(complex_gt)
            simp_comp['simple_guess'] = simple_guess
            simp_comp['complex_guess'] = complex_guess

        return simp_comp
    
    
def main(**kwargs):
    test_loader = kwargs.get('test_loader')
    isVQAeval = kwargs.get('isVQAeval')
    logger = kwargs.get('logger')    
    ds = kwargs.get('dsname')
    jsonfolder = kwargs.get('jsonfolder')    
    testset = test_loader.dataset.data
    
    json_paths ={}
    json_paths['Mutan'] = os.path.join(jsonfolder,ds,'mutan')
    json_paths['Zhang'] = os.path.join(jsonfolder,ds,'zhang')
    json_paths['UpDown'] = os.path.join(jsonfolder,ds,'updown')

    format_acc_rmse = "\tRMSE:{:.2f} Accuracy {:.2f}%"
    format_acc_rmse_sc = "\t{} RMSE:{:.2f} Accuracy {:.2f}%"

    detect_gt,predictions = get_detect(testset)
    logger.write("Detect:")
    
    if isVQAeval:          
        acc,rmse = evalvqa(testset,predictions,isVQAeval)
        logger.write(format_acc_rmse.format(rmse,acc))         
        HQAjson_path = os.path.join(jsonfolder,'results_package.json')
        for i in [0,1,2,'IRLC']:
            key = str(i)
            logger.write("Guess {}".format(i))   
            VQA_acc, rmse = evalHQA(key,testset,jsonpath=HQAjson_path)
            logger.write(format_acc_rmse.format(rmse,VQA_acc))


        for method in ['Mutan','Zhang']:
            logger.write(method)
            jsonpath  = load_folder(json_paths[method],"json")
            predictions = eval_zhang_updown_mutan(testset,jsonpath)                    
            acc,rmse = evalvqa(testset,predictions,isVQAeval)
            logger.write(format_acc_rmse.format(rmse,acc))
        
        #from interpretable paper
        logger.write("UpDown:(from IRLC paper)")
        logger.write(format_acc_rmse.format(2.69,51.5))

        
    else:
        simp_comp = eval_simp_comp(testset,predictions,baselines=True)
        for d in ['simple','complex']:
            acc,rmse = simp_comp[d]
            logger.write(format_acc_rmse_sc.format(d,rmse,acc))
            
        for i in [0,1,2]:  
            logger.write("Guess {}".format(i)) 
            for d in ['simple','complex']:
                guess = simp_comp[d+"_guess"]
                acc,rmse = guess[i]                             
                logger.write(format_acc_rmse_sc.format(d,rmse,acc))
        
   
        for method in json_paths:
            logger.write(method)
            jsonpath  = load_folder(json_paths[method],"json")
            predictions = eval_zhang_updown_mutan(testset,jsonpath)
            simp_comp = eval_simp_comp(testset,predictions)
            for d in ['simple','complex']:
                acc,rmse = simp_comp[d]
                logger.write(format_acc_rmse_sc.format(d,rmse,acc))


