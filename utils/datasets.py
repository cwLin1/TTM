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
test_tfm = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_tfm = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class Ego4D(Dataset):
    def __init__(self, data_dir, tfm, split, subset = 'test'):
        super(Ego4D).__init__()
        self.video_dir = os.path.join(data_dir, 'videos')
        self.audio_dir = os.path.join(data_dir, 'audios')
        self.seg_dir = os.path.join(data_dir, 'train', 'seg')
        self.bbox_dir = os.path.join(data_dir, 'train', 'bbox')
        
        self.video_ids = sorted([x.split('_')[0] for x in os.listdir(self.seg_dir)])
        if subset == 'train':
            self.video_ids = self.video_ids[:split]
        else:
            self.video_ids = self.video_ids[split:]
        
        self.seg = []
        self.bbox = {}
        for video_id in self.video_ids:
            seg_path = os.path.join(self.seg_dir, video_id + '_seg.csv')
            bbox_path = os.path.join(self.bbox_dir, video_id + '_bbox.csv')
            with open(seg_path, newline='') as csvfile:
                rows = csv.DictReader(csvfile)
                for row in rows:
                    row['video_id'] = video_id
                    self.seg.append(row)

            with open(bbox_path, newline='') as csvfile:
                rows = csv.DictReader(csvfile)
                self.bbox[video_id] = []
                for row in rows:
                    self.bbox[video_id].append(row)

        self.transform = tfm
    
    def get_bbox(self, bbox_dict, person_id, start_frame, end_frame):
        bbox = []
        for row in bbox_dict:
            if row['person_id'] == person_id and int(row['frame_id']) >= start_frame and int(row['frame_id']) < end_frame:
                bbox.append([int(np.floor(float(row['x1']))), int(np.floor(float(row['x2']))), 
                            int(np.ceil(float(row['y1']))), int(np.ceil(float(row['y2'])))])
        return bbox

    def __len__(self):
        return len(self.seg)
  
    def __getitem__(self, idx):
        seg_id = self.seg[idx]
        print(seg_id)
        video_id = seg_id['video_id']
        person_id = seg_id['person_id']
        start_frame = int(seg_id['start_frame'])
        end_frame = int(seg_id['end_frame'])
        label = float(seg_id['ttm'])

        video_path = os.path.join(self.video_dir, video_id + '.mp4')
        audio_path = os.path.join(self.audio_dir, video_id + '.wav')

        cap = cv2.VideoCapture(video_path)
        cap.set(1, start_frame)
        frame_num = start_frame

        segment = torch.zeros(0, 3, 224, 224)
        ret, frame = cap.read() # Read the frame
        bbox = self.get_bbox(self.bbox[video_id], person_id, start_frame, end_frame)
        while ret and frame_num < end_frame:
            x1, x2, y1, y2 = bbox[frame_num-start_frame]
            if x1 == -1 and x2 == -1 and y1 == -1 and y2 == -1:
                image = np.zeros((224,224,3)).astype('uint8')
            else:
                image = frame[y1:y2, x1:x2]
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            image = self.transform(image).unsqueeze(0)
            segment = torch.cat((segment, image), 0)

            ret, frame = cap.read()
            frame_num += 1

        ori_audio, ori_sample_rate = torchaudio.load(audio_path, normalize = True)
        sample_rate = 16000
        transform = torchaudio.transforms.Resample(ori_sample_rate, sample_rate)
        audio = transform(ori_audio)
        # print(audio.size())

        onset = int(start_frame/30 * sample_rate)
        offset = int(end_frame/30 * sample_rate)
        crop_audio = audio[:, onset:offset]

        return segment, crop_audio, label

def get_train_val_loader(path, split):
    num_train = len(os.listdir(os.path.join(path, 'train', 'seg')))
    valid_slice = int(num_train * split)

    data_train = Ego4D(path, train_tfm, valid_slice, 'train')
    data_val = Ego4D(path, test_tfm, valid_slice, 'test')

    return data_train, data_val



if __name__ == '__main__':
    data_path = "dlcv-final-problem1-talking-to-me/student_data/student_data"
    train, val = get_train_val_loader(data_path, 0.8)
    print(len(train))
    seg, audio, label = train[6]
    print(seg.size())
    print(audio.size())