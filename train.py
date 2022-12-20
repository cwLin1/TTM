import math
import numpy as np
import csv
from tqdm import tqdm

import torch 
import torch.nn as nn

from models.model import load_model
from utils.datasets import get_train_val_loader
from torch.utils.data import Dataset, DataLoader, random_split

# def form_slice(batch_size, length):
#     return [[i*batch_size, min((i+1)*batch_size, length)] for i in range(int(length//batch_size)+1)]

device = ("cuda" if torch.cuda.is_available() else "cpu")

data_path = "dlcv-final-problem1-talking-to-me/student_data/student_data"
train_set, val_set = get_train_val_loader(data_path, 0.8)

model = load_model(1024+13680).to(device)

n_epochs, best_acc = 320, 0

criterion = nn.MSELoss(reduction='mean')
# criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)

loss_train = []
acc_train = []
loss_val = []
acc_val = []

# train_slice = form_slice(64, len(train_set))

for epoch in range(n_epochs):
    model.train()
    loss_record = []
    acc_record = []

    # for start, end in tqdm(train_slice):
    #     x_video_batch = []
    #     x_audio_batch = []
    #     labels_batch = []

    #     for idx in range(start, end):
    #         x_video, x_audio, label = train_set[idx]
    #         x_video = x_video.to(device)        # Move your data to device.
    #         x_audio = x_audio.to(device)
    #         b_size = x_video.size(0)
    #         labels = (label * torch.ones(b_size)).to(torch.int64).to(device)
    #         x_video_batch.append(x_video)
    #         x_audio_batch.append(x_audio)
    #         labels_batch.append(labels)
    #     x_video_batch = torch.cat(x_video_batch)
    #     x_audio_batch = torch.cat(x_audio_batch)
    #     labels_batch = torch.cat(labels_batch)
    #     pred = model(x_video_batch, x_audio_batch)
    #     loss = criterion(pred, labels_batch)
    #     loss.backward()                     # Compute gradient(backpropagation).
    #     optimizer.step()                    # Update parameters.

    #     loss_record.append(loss.detach().item())


    for x_video, x_audio, label in tqdm(train_set):
        b_size = x_video.size(0)
        optimizer.zero_grad()               # Set gradient to zero.
        x_video = x_video.to(device)        # Move your data to device.
        x_audio = x_audio.to(device)
        labels = (label * torch.ones(b_size)).to(torch.float).to(device)

        pred = model(x_video, x_audio)
        # print(pred)
        loss = criterion(pred, labels)
        loss.backward()                     # Compute gradient(backpropagation).
        optimizer.step()                    # Update parameters.

        loss_record.append(loss.detach().item())

        # acc = pred.sum(0).argmax() == label
        acc = (pred.sum() / len(pred) > 0.5) == label
        acc_record.append(acc)
    
    loss_train = sum(loss_record)/len(loss_record)
    acc_train = sum(acc_record)/len(acc_record)
    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {loss_train:.5f}, acc = {acc_train:.5f}")

    # ---------- Validation ----------
    loss_record = []
    acc_record = []
    for x_video, x_audio, label in tqdm(val_set):
        b_size = x_video.size(0)
        x_video = x_video.to(device)        # Move your data to device.
        x_audio = x_audio.to(device)

        with torch.no_grad():
            pred = model(x_video, x_audio)

        labels = (label * torch.ones(b_size)).to(torch.float).to(device)
        loss = criterion(pred, labels)

        loss_record.append(loss.detach().item())

        # acc = pred.sum(0).argmax() == label
        acc = (pred.sum() / len(pred) > 0.5) == label
        acc_record.append(acc)

    loss_val = sum(loss_record)/len(loss_record)
    acc_val = sum(acc_record)/len(acc_record)
    print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {loss_val:.5f}, acc = {acc_val:.5f}")

    if acc_val > best_acc:
        torch.save(model.state_dict(), "ckpts/best.ckpt")
        best_acc = acc_val
        
    print("Best acc:", best_acc)
