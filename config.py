from collections import defaultdict
from models.baseline import Qmodel,Imodel,QImodel
from models import RN_NAC,RN_GTU,RN_BGOG,RN_GTU_norm,RN_OG_embd
dataset = defaultdict(list)

#global config
global_config = {}
global_config['jsonfolder'] ='/home/manoj/mutan'
global_config['coco_pool_features'] = '/hdd/manoj/IMGFEATS/resnet152.h5'
global_config['genome_pool_features'] = '/hdd/manoj/IMGFEATS/resnet152_genome.h5'
global_config['coco_bottomup'] = '/home/manoj/bottomup_1_100/ssd/genome-trainval.h5'
global_config['cocotest_bottomup'] = '/home/manoj/bottomup_1_100/ssd/genome-test.h5'
global_config['genome_bottomup'] = '/home/manoj/bottomup_1_100/ssd/genome_ourdb/genome-trainval.h5'


#dictionary

global_config['dictionaryfile'] = 'data/dictionary.pickle'
global_config['glove'] = 'data/glove6b_init_300d.npy'


#dataset configs
name= 'Ourdb'
dataset[name] = {}
dataset[name]['N_classes'] = 16
dataset[name]['test'] = '/home/manoj/Downloads/counting/data/test_data_kushalformat_fixed.pkl'
dataset[name]['train'] = '/home/manoj/Downloads/counting/data/train_data_kushalformat_fixed.pkl'

name= 'HowmanyQA'
dataset[name] = {}
dataset[name]['N_classes'] = 21
dataset[name]['test'] = '/home/manoj/Downloads/HowMany-QA/howmanyQA_test.pkl'
dataset[name]['train'] = '/home/manoj/Downloads/HowMany-QA/howmanyQA_train.pkl'

name= 'VQA2'
dataset[name] = {}
dataset[name]['N_classes'] = 21
dataset[name]['train'] = '/home/manoj/Videos/nn/utils/VQA/count_trainval_vqa2.0.pickle'
dataset[name]['testdev'] = '/home/manoj/Videos/nn/utils/VQA/count_testdev_vqa2.0.pickle'
dataset[name]['teststd'] = '/home/manoj/Videos/nn/utils/VQA/count_teststd_vqa2.0.pickle'

#model names
models = { 'Q':Qmodel, 'I': Imodel, 'QI': QImodel ,
          'RN_BGOG': RN_BGOG.RN,
          'RN_NAC': RN_NAC.RN,
          'RN_GTU': RN_GTU.RN,
          'RN_OG_embd': RN_OG_embd.RN,
          'RN_GTU_norm': RN_GTU_norm.RN,
          } 
