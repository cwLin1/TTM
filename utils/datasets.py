import numpy as np
import torch
import os
import torchvision.transforms as transforms
from PIL import Image
import cv2
import csv
from torch.utils.data import DataLoader, Dataset
import torchaudio
from tqdm import tqdm
#====Transforms====
class Ego4D(Dataset):
    def __init__(self, data_dir, split, subset = 'test', tfm = None):
        super(Ego4D).__init__()
        self.subset = subset
        self.video_dir = os.path.join(data_dir, 'videos')
        self.audio_dir = os.path.join(data_dir, 'audios')
        self.video_feature_dir = os.path.join(data_dir, 'r21d')
        self.audio_feature_dir = os.path.join(data_dir, 'vggish')
        if self.subset == 'val' or self.subset == 'train':
            self.seg_dir = os.path.join(data_dir, 'train', 'seg')
        else:
            self.seg_dir = os.path.join(data_dir, 'test', 'seg')

        self.video_ids = sorted([x.split('_')[0] for x in os.listdir(self.seg_dir)])

        n_data = len(self.video_ids)
        fold = 0
        
        val_onset = int((1-split) * fold * n_data)
        val_offset = int((1-split) * (fold+1) * n_data)

        if self.subset == 'train':
            self.video_ids = self.video_ids[:val_onset] + self.video_ids[val_offset:]
        else:
            self.video_ids = self.video_ids[val_onset:val_offset]
        
        self.seg = []
        self.bbox = {}
        for video_id in self.video_ids:
            seg_path = os.path.join(self.seg_dir, video_id + '_seg.csv')
            with open(seg_path, newline='') as csvfile:
                rows = csv.DictReader(csvfile)
                for row in rows:
                    row['video_id'] = video_id
                    self.seg.append(row)

        self.transform = tfm

        self.video_features = []
        self.audio_features = []
        self.labels = []

        print('Loading data...')
        for seg_id in tqdm(self.seg):
            video_id = seg_id['video_id']
            person_id = seg_id['person_id']
            start_frame = int(seg_id['start_frame'])
            end_frame = int(seg_id['end_frame'])
            feature_id = video_id + '_' + person_id + '_' + str(start_frame) + '_' + str(end_frame)

            if self.subset == 'test':
                label = feature_id
            else:
                label = float(seg_id['ttm'])

            video_feature_path = os.path.join(self.video_feature_dir, feature_id + '.npy')
            audio_feature_path = os.path.join(self.audio_feature_dir, feature_id + '.npy')
            video_feature = np.load(video_feature_path)
            video_feature = torch.from_numpy(video_feature).float()

            audio_feature = np.load(audio_feature_path)
            audio_feature = torch.from_numpy(audio_feature).float()

            if video_feature.shape[0] > audio_feature.shape[0]:
                video_feature = video_feature[:audio_feature.shape[0]]
            
            self.video_features.append(video_feature)
            self.audio_features.append(audio_feature)
            self.labels.append(label)

    def __len__(self):
        return len(self.seg)
  
    def __getitem__(self, idx):
        # ======
        # video_feature: n x 1024
        # audio_feature: n x 2 x 40 x 171
        # ======
        return self.video_features[idx], self.audio_features[idx], self.labels[idx]

def get_train_val_loader(path, split, subset='train'):
    if subset == 'train':
        data_train = Ego4D(path, split, 'train')
        data_val = Ego4D(path, split, 'val')
        return data_train, data_val
    else:
        data_test = Ego4D(path, 0, 'test')
        return data_test
    

if __name__ == '__main__':
    data_path = "dlcv-final-problem1-talking-to-me/student_data/student_data"

    train_set, val_set = get_train_val_loader(data_path, 0.8)
    # print(len(train_set))

    video, audio, label = train_set[6]
    print(video)
    print(audio)