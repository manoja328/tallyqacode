from collections import defaultdict
dataset = defaultdict(list)

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

