import math
import numpy as np
import csv
from tqdm import tqdm

import torch 
import torch.nn as nn

from models.model import load_model
from utils.datasets import get_train_val_loader
from torch.utils.data import Dataset, DataLoader, random_split

device = ("cuda" if torch.cuda.is_available() else "cpu")

data_path = "dlcv-final-problem1-talking-to-me/student_data/student_data"
test_set = get_train_val_loader(data_path, 0, 'test')

model_path = "ckpts/best.ckpt"
model = load_model(1024+13680, model_path).to(device)
model.()

output_path = '1.csv'
file = open(output_path, 'w')
writer = csv.writer(file)
writer.writerow(['Id', 'Predicted'])
    

prediction = []
for x_video, x_audio, id in tqdm(test_set):
    b_size = x_video.size(0)
    x_video = x_video.to(device)
    x_audio = x_audio.to(device)

    with torch.no_grad():
        pred = model(x_video, x_audio)

    writer = csv.writer(file)
    writer.writerow([id, int((pred.sum() / len(pred) > 0.5).item())])
