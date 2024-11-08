<<<<<<< HEAD
import json
from PIL import Image
from torch.utils.data import Dataset
import os

class Sarcasm(Dataset):
    def __init__(self, annotation = 'annotations', file_annotation = 'vimmsd_train.json', file_image = 'train-images'):
        super().__init__()
        self.__data = {}
        
        # Path of folder images and file annotations
        self.file_json = os.path.join('.', annotation, file_annotation)
        self.file_image = os.path.join('.', file_image)
        
        with open(self.file_json, 'r', encoding = 'utf-8') as file:
            data_json = json.load(file)
        for i_th, (idx, value) in enumerate(data_json.items()):
            image_path = os.path.join(self.file_image, value['image'])
            img = Image.open(image_path).convert("RGB")
            
            self.__data[i_th] = {
                'id': idx,
                'image': img,
                'caption': value['caption'],
                'label': value['label']
            }
            
    def __len__(self):
        return len(self.__data)
    
    def __getitem__(self, index):
        return self.__data[index]
        
=======
import json
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from data_utils.utils import preprocessing_label
import os

class Sarcasm(Dataset):
    def __init__(self, data_dir = "data", file_annotation = 'vimmsd_train.json', file_image = 'train-images'):
        super().__init__()
        self.__data = {}
        
        # Path of folder images and file annotations
        self.file_json = os.path.join('.', data_dir, file_annotation)
        self.file_image = os.path.join('.', data_dir, file_image)
        
        with open(self.file_json, 'r', encoding = 'utf-8') as file:
            data_json = json.load(file)
        for i_th, (idx, value) in enumerate(data_json.items()):
            image_path = os.path.join(self.file_image, value['image'])
            img = Image.open(image_path).convert("RGB")
            
            self.__data[i_th] = {
                'id': idx,
                'image': np.array(img),
                'caption': value['caption'],
                'label': preprocessing_label(value['label'])
            }
            
    def __len__(self):
        return len(self.__data)
    
    def __getitem__(self, index):
        return self.__data[index]
>>>>>>> 7eb23a3c720d02f8ed4357e6bcb110787aa71cce
