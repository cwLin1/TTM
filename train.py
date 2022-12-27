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
train_set, val_set = get_train_val_loader(data_path, 0.8)
output_dim = 1

# model_path = "ckpts/last.ckpt"
model_path = None
model = load_model(1024, 384, output_dim, model_path).to(device)

n_epochs = 80
best_acc = 0

if output_dim == 1:
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
loss_train = []
acc_train = []
loss_val = []
acc_val = []

# train_slice = form_slice(64, len(train_set))

for epoch in range(0, n_epochs):
    model.train()
    loss_record = []
    acc_record = []

    shuffle = np.random.permutation(len(train_set))
    for i in tqdm(range(len(train_set))):
        x_video, x_audio, label = train_set[shuffle[i]]
        b_size = x_video.size(0)
        optimizer.zero_grad()               # Set gradient to zero.
        x_video = x_video.to(device)        # Move your data to device.
        x_audio = x_audio.to(device)
        labels = (label * torch.ones(b_size)).to(torch.float).to(device)

        pred = model(x_video, x_audio)
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
    if len(val_set) > 0:
        loss_record = []
        acc_record = []
        model.eval()
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

        if acc_val.item() > best_acc:
            torch.save(model.state_dict(), "ckpts/best.ckpt")
            best_acc = acc_val.item()
        print("Best acc:", best_acc)

    torch.save(model.state_dict(), "ckpts/last.ckpt")

