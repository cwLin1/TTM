import numpy as np
import torch
import os
import torchvision.transforms as transforms
from PIL import Image
import cv2
import csv
from torch.utils.data import DataLoader, Dataset
import torchaudio

#====Transforms====
class Ego4D(Dataset):
    def __init__(self, data_dir, split, subset = 'test', tfm = None):
        super(Ego4D).__init__()
        self.subset = subset
        self.video_dir = os.path.join(data_dir, 'videos')
        self.audio_dir = os.path.join(data_dir, 'audios')
        self.video_feature_dir = os.path.join(data_dir, 'i3d')
        self.MFCC_dir = os.path.join(data_dir, 'MFCC')
        if self.subset == 'val' or self.subset == 'train':
            self.seg_dir = os.path.join(data_dir, 'train', 'seg')
        else:
            self.seg_dir = os.path.join(data_dir, 'test', 'seg')

        self.video_ids = sorted([x.split('_')[0] for x in os.listdir(self.seg_dir)])
        if self.subset == 'train':
            self.video_ids = self.video_ids[:split]
        else:
            self.video_ids = self.video_ids[split:]
        
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

    def __len__(self):
        return len(self.seg)
  
    def __getitem__(self, idx):
        seg_id = self.seg[idx]
        # print(seg_id)
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
        audio_feature_path = os.path.join(self.MFCC_dir, feature_id + '.npy')

        if video_feature.shape[0] > audio_feature.shape[0]:
            video_feature = video_feature[:audio_feature.shape[0]]
        
        video_feature = np.load(video_feature_path)
        video_feature = torch.from_numpy(video_feature).float()

        audio_feature = np.load(audio_feature_path)
        audio_feature = torch.from_numpy(audio_feature).float()

        # ori_audio, ori_sample_rate = torchaudio.load(audio_path, normalize = True)
        # sample_rate = 16000
        # audio_resample = torchaudio.transforms.Resample(ori_sample_rate, sample_rate)
        # audio = audio_resample(ori_audio)
        # print(audio.size())

        # onset = int(start_frame/30 * sample_rate)
        # offset = int(end_frame/30 * sample_rate)

        # crop_audio = torch.zeros(2, 34134 * max(1, int(np.ceil((offset-onset)/34134))))
        # crop_audio[:, 0:offset - onset] = audio[:, onset:offset]
        # crop_audio = crop_audio.view(max(1, int(np.ceil((offset-onset)/34134))), 2, 34134)
        # crop_audio = audio[:, onset:offset]
        # print(crop_audio.size())
        # audio_transform = torchaudio.transforms.MFCC()
        # audio_feature = audio_transform(crop_audio)

        # ======
        # video_feature: n x 1024
        # audio_feature: n x 2 x 40 x 171
        # ======
        if self.subset == 'test':
            return video_feature, audio_feature, feature_id
        return video_feature, audio_feature, label

def get_train_val_loader(path, split, subset='train'):
    num_train = len(os.listdir(os.path.join(path, subset, 'seg')))
    valid_slice = int(num_train * split)
    if subset == 'train':
        data_train = Ego4D(path, valid_slice, 'train')
        data_val = Ego4D(path, valid_slice, 'val')
        return data_train, data_val
    else:
        data_test = Ego4D(path, 0, 'test')
        return data_test
    

if __name__ == '__main__':
    data_path = "dlcv-final-problem1-talking-to-me/student_data/student_data"

    train_set, val_set = get_train_val_loader(data_path, 1)
    # print(len(train_set))

    seg, audio, label = train_set[6]
    print(seg)
    # print(audio)
