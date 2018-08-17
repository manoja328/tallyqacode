from collections import defaultdict
dataset = defaultdict(list)

#global config
global_config = {}
global_config['jsonfolder'] ='/home/manoj/mutan'
global_config['coco_pool_features'] = '/media/manoj/hdd/IMGFEATS/resnet152.h5'
global_config['genome_pool_features'] = '/media/manoj/hdd/IMGFEATS/resnet152_genome.h5'
global_config['coco_bottomup'] = '/home/manoj/bottomup_1_100/ssd/genome-trainval.h5'
global_config['genome_bottomup'] = '/home/manoj/bottomup_1_100/ssd/genome_ourdb/genome-trainval.h5'

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
