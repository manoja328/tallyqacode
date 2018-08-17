import argparse
import os
import utils
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dsname', help='dataset: Ourdb | HowmanyQA' , default='HowmanyQA')
    parser.add_argument('--model', help='Model Q | I| QI | Main | RN',default='Q')
    parser.add_argument('--save', help='save folder name',default='01')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = parse_args()
       
    isVQAeval = False
    if args.dsname == 'HowmanyQA':
        isVQAeval = True

    savefolder = '_'.join([args.dsname,args.model,args.save])
    logger = os.path.join(savefolder, 'log.txt')
    
    info_json = os.path.join(savefolder,'infos.json')
    js = utils.parsejson(info_json)
    
    for key in js:
        plt.plot(js[key],label=key)
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()
