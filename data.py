from torch.utils.data import Dataset
import numpy as np
import torch
from models.language import getglove
import pickle
import h5py

class CountDataset(Dataset):

    def __init__(self,**kwargs):

        
        file = kwargs.get('file')
        
        with open(file,'rb') as f:
            self.data = pickle.load(f)
            
#        self.data = self.data[:32]
#        self.dictionary = dictionary
             
        self.pool_features_path_coco = kwargs.get('coco_pool_features')
        self.pool_features_path_genome = kwargs.get('genome_pool_features')
        self.poolcoco_id_to_index =  self._poolcreate_coco_id_to_index(self.pool_features_path_coco)
        self.poolcoco_id_to_index_gen =  self._poolcreate_coco_id_to_index(self.pool_features_path_genome)        

        self.image_features_path_coco = kwargs.get('coco_bottomup')
        self.coco_id_to_index =  self.id_to_index(self.image_features_path_coco)  
        self.image_features_path_genome = kwargs.get('genome_bottomup')
        self.genome_id_to_index =  self.id_to_index(self.image_features_path_genome)        

    def id_to_index(self,path):
        """ Create a mapping from a COCO image id into the corresponding index into the h5 file """
               
        with  h5py.File(path, 'r') as features_file:
            coco_ids = features_file['ids'][:]
        coco_id_to_index = {name: i for i, name in enumerate(coco_ids)}
        return coco_id_to_index       
        
    
    def _load_image_coco(self, image_id):
        """ Load an image """
        if not hasattr(self, 'features_file'):
            # Loading the h5 file has to be done here and not in __init__ because when the DataLoader
            # forks for multiple works, every child would use the same file object and fail
            # Having multiple readers using different file objects is fine though, so we just init in here.
            self.features_file = h5py.File(self.image_features_path_coco, 'r')
           
        index = self.coco_id_to_index[image_id]
        L = self.features_file['num_boxes'][index]
        W = self.features_file['widths'][index]
        H = self.features_file['heights'][index]
        box_feats = self.features_file['features'][index]
        box_locations = self.features_file['boxes'][index]
        # find the boxes with all co-ordinates 0,0,0,0
        #L = np.where(~box_locations.any(axis=1))[0][0]
        
        return L,W,H,box_feats.T,box_locations.T
 

    def _load_image_genome(self, image_id):
        """ Load an image """
        if not hasattr(self, 'features_file_genome'):
            # Loading the h5 file has to be done here and not in __init__ because when the DataLoader
            # forks for multiple works, every child would use the same file object and fail
            # Having multiple readers using different file objects is fine though, so we just init in here.
            self.features_file_genome = h5py.File(self.image_features_path_genome, 'r')
           
        image_id = int(str(image_id)[1:])
        index = self.genome_id_to_index[image_id]
        L = self.features_file_genome['num_boxes'][index]
        W = self.features_file_genome['widths'][index]
        H = self.features_file_genome['heights'][index]
        box_feats = self.features_file_genome['features'][index]
        box_locations = self.features_file_genome['boxes'][index]
        # find the boxes with all co-ordinates 0,0,0,0
        #L = np.where(~box_locations.any(axis=1))[0][0]
        
        return L,W,H,box_feats.T,box_locations.T        
        
    def _poolcreate_coco_id_to_index(self , path):
        """ Create a mapping from a COCO image id into the corresponding index into the h5 file """
        with h5py.File(path, 'r') as features_file:
            coco_ids = features_file['filenames'][()]
        coco_id_to_index = {name: i for i, name in enumerate(coco_ids)}
        return coco_id_to_index        
    
   

          
    def _load_pool_image(self, image_id):
        """ Load an image """
        if not hasattr(self, 'pool_file'):
            # Loading the h5 file has to be done here and not in __init__ because when the DataLoader
            # forks for multiple works, every child would use the same file object and fail
            # Having multiple readers using different file objects is fine though, so we just init in here.
            self.pool_file = h5py.File(self.pool_features_path_coco, 'r')
        if not hasattr(self, 'pool_file_gen'):
            # Loading the h5 file has to be done here and not in __init__ because when the DataLoader
            # forks for multiple works, every child would use the same file object and fail
            # Having multiple readers using different file objects is fine though, so we just init in here.
            self.pool_file_gen = h5py.File(self.pool_features_path_genome, 'r')          
            
        index = self.poolcoco_id_to_index.get(image_id,None)
        if index is not None:
            pooled = self.pool_file['pool5'][index]        
            unpooled = self.pool_file['res5c'][index]
            return torch.from_numpy(pooled).float(), torch.from_numpy(unpooled).float()  
        else:
            index = self.poolcoco_id_to_index_gen.get(image_id,None)
            pooled = self.pool_file_gen['pool5'][index]        
            unpooled = self.pool_file_gen['res5c'][index]
            return torch.from_numpy(pooled).float(), torch.from_numpy(unpooled).float()    

    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ent = self.data[idx]
        qid = ent['question_id']
        img_name = ent['image']
        img_id = ent['image_id']
        ans = ent.get('multiple_choice_answer',None)
        if ans is None:
            ans = ent['answer']
        que = ent['question']
       
        lasttwo = '/'.join(img_name.split("/")[-2:])
        lasttwo +=".pkl"
        wholefeat,pooled = self._load_pool_image(lasttwo[:-4])

        lastone = lasttwo.split("/")[-1]
        
#        pk = pickle.load(open(os.path.join("/home/manoj/448feats/feats",lastone),"rb"))
#        L =  len(pk) - 1 # lenght of entries in pickle file

        if 'VG' in img_name:
            L, W, H ,imgarr,box_coords = self._load_image_genome(img_id)
        else:
            L, W, H ,imgarr,box_coords = self._load_image_coco(img_id)
        
#        tokens = tokenize_ques(self.dictionary,"How many dogs?")
#        qfeat = torch.from_numpy(tokens)
        
        qfeat = getglove(que)
        qfeat = torch.from_numpy(qfeat)

        imgarr = torch.from_numpy(imgarr)
        box_coords = torch.from_numpy(np.array(box_coords,dtype=np.float32))        
        scale = torch.from_numpy(np.array([W,H,W,H],dtype=np.float32))
        box_coords = box_coords / scale   
        return qid,wholefeat,pooled,imgarr.float(),np.float32(ans),qfeat,box_coords.float(),L

