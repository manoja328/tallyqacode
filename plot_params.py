import argparse
import os
import utils
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', help='save folder name',default='01')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = parse_args()
       
    savefolder = args.save
    
    info_json = os.path.join(savefolder,'infos.json')
    js = utils.parsejson(info_json)
    
    for key in js:
        L = len(js[key])
        plt.plot(range(L),js[key],'o')
        plt.plot(range(L),js[key],label=key)
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()
