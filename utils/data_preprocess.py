import os
import cv2
import csv
import numpy as np

from moviepy.editor import VideoFileClip
import torchaudio
import torch
from tqdm import tqdm

def audio_transform(data_dir):
    video_dir = os.path.join(data_dir, 'videos')
    audio_dir = os.path.join(data_dir, 'audios')
    os.makedirs(audio_dir, exist_ok=True)
    
    videos = sorted([os.path.join(video_dir, x) for x in os.listdir(video_dir) if x.endswith(".mp4")])

    for video_path in videos:
        video_id = video_path.split('/')[-1][:-4]
        video = VideoFileClip(video_path)
        audio = video.audio
        # audio.write_audiofile(os.path.join(audio_dir, video_id + '.wav'))

def audio_feature(data_dir):
    sets = ['test']
    for subset in sets:
        seg_dir = os.path.join(data_dir, subset, 'seg')
        video_ids = sorted([x.split('_')[0] for x in os.listdir(seg_dir)])
        seg = []
        
        for video_id in video_ids:
            seg_path = os.path.join(seg_dir, video_id + '_seg.csv')
            with open(seg_path, newline='') as csvfile:
                rows = csv.DictReader(csvfile)
                for row in rows:
                    row['video_id'] = video_id
                    seg.append(row)

        audio_dir = os.path.join(data_dir, 'audios')
        feature_dir = os.path.join(data_dir, 'MFCC')
        os.makedirs(feature_dir, exist_ok=True)
        
        for seg_id in tqdm(seg):
            video_id = seg_id['video_id']
            start_frame = int(seg_id['start_frame'])
            end_frame = int(seg_id['end_frame'])

            audio_path = os.path.join(audio_dir, video_id + '.wav')
            ori_audio, ori_sample_rate = torchaudio.load(audio_path, normalize = True)
            sample_rate = 16000
            audio_resample = torchaudio.transforms.Resample(ori_sample_rate, sample_rate)
            audio = audio_resample(ori_audio)

            onset = int(start_frame/30 * sample_rate)
            offset = int(end_frame/30 * sample_rate)
            crop_audio = torch.zeros(2, 34134 * max(1, int(np.ceil((offset-onset)/34134))))
            crop_audio[:, 0:offset - onset] = audio[:, onset:offset]
            crop_audio = crop_audio.view(max(1, int(np.ceil((offset-onset)/34134))), 2, 34134)

            audio_transform = torchaudio.transforms.MFCC()
            audio_feature = audio_transform(crop_audio).numpy()
            save_path = os.path.join(data_dir, 'MFCC', seg_id['video_id'] + '_' + seg_id['person_id'] + '_' + seg_id['start_frame'] + '_' + seg_id['end_frame'] + '.npy')

            np.save(save_path, audio_feature)


def video_transform(data_path):
    video_dir = os.path.join(data_path, 'videos')
    frame_dir = os.path.join(data_path, 'frames')
    os.makedirs(frame_dir, exist_ok=True)
    
    videos = sorted([os.path.join(video_dir, x) for x in os.listdir(video_dir) if x.endswith(".mp4")])

    for video_path in videos[:1]:
        video_id = video_path.split('/')[-1][:-4]
        print(video_id)
        os.makedirs(os.path.join(frame_dir, video_id), exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        success, image = cap.read()
        imgs = []
        frame_num = 0
        while success:
            # cv2.imwrite(os.path.join(frame_dir, video_id, "%05d.jpg" % frame_num), image)
            success, image = cap.read()
            imgs.append(image)
            frame_num += 1
        print('finish')

def read_clip(data_dir, start_frame, end_frame):
    video_dir = os.path.join(data_dir, 'videos')
    
    videos = sorted([os.path.join(video_dir, x) for x in os.listdir(video_dir) if x.endswith(".mp4")])
    video_path = videos[0]
    cap = cv2.VideoCapture(video_path)
    cap.set(1, start_frame); # Where frame_no is the frame you want

    ret, frame = cap.read() # Read the frame
    frame_num = 0
    while ret and frame_num < end_frame - start_frame:
        ret, frame = cap.read()
        frame_num += 1

if __name__ == '__main__':
    data_dir = "dlcv-final-problem1-talking-to-me/student_data/student_data"
    # audio_transform(data_dir)
    audio_feature(data_dir)