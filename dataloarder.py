import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random
import glob
from PIL import Image, ImageFilter

class classification(Dataset):
    
    def __init__(self, is_train, transform=None):
        super(classification, self).__init__()
        self.is_train = is_train
        self.transform = transform
        if is_train == 1:
            # 해당 폴더 안의 데이터 파일 결로를 리스트에 담아두는 과정이 필요
            self.data_list = glob.glob('./data/emotion_kaggle/train/*/*')
            self.label_list = os.listdir('./data/emotion_kaggle/train/')
        else:
            self.data_list = glob.glob('./data/emotion_kaggle/test/*/*')
            self.label_list = os.listdir('./data/emotion_kaggle/test/')

    # 학습 데이터의 개수를 리턴    
    def __len__(self):
        return len(self.data_list)

    # 앞서 만든 리스트의 인덱스 값을 참조해 해당 이미지를 연 다음 이미지 전처리 실행 
    def __getitem__(self, idx):
        path = self.data_list[idx]
        label = self.data_list[idx].split('/')[4] 
        label_idx = self.label_list.index(label)
#         img_path = self.data_list[idx].split('train/')[-1]
        image = Image.open(path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return image, label_idx #, img_path
