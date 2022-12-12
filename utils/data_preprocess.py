import os
import cv2

from moviepy.editor import VideoFileClip

def audio_transform(data_path):
    video_dir = os.path.join(data_path, 'videos')
    audio_dir = os.path.join(data_path, 'audios')
    os.makedirs(audio_dir, exist_ok=True)
    
    videos = sorted([os.path.join(video_dir, x) for x in os.listdir(video_dir) if x.endswith(".mp4")])

    for video_path in videos:
        video_id = video_path.split('/')[-1][:-4]
        video = VideoFileClip(video_path)
        audio = video.audio
        # audio.write_audiofile(os.path.join(audio_dir, video_id + '.wav'))

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

def read_clip(data_path, start_frame, end_frame):
    video_dir = os.path.join(data_path, 'videos')
    
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
    data_path = "dlcv-final-problem1-talking-to-me/student_data/student_data"
    audio_transform(data_path)
    # read_clip(data_path, 300, 400)