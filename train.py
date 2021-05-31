import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import math
import time
import argparse
from model import SNIPER
from params import nyc_params, chicago_params
from lib.utils import get_neigh_index, prepare_data, loss_function

parser = argparse.ArgumentParser()
parser.add_argument("--gpus", type=str, help="test program")
parser.add_argument("--dataset", type=str, help="test program")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataset = args.dataset
if dataset == 'nyc':
    params = nyc_params
elif dataset == 'chicago':
    params = chicago_params
else:
    raise NameError

dr = params.dr
len_recent_time = params.len_recent_time
number_sp = params.number_sp
number_region = params.number_region

thre_nc = dataset + '/' + params.threshold_nc
label = dataset + '/' + params.label
all_data = dataset + '/' + params.all_data
label = torch.tensor(np.load(file=label)[len_recent_time:]).to(device)
all_data = prepare_data(np.load(file=all_data), len_recent_time).to(device)
thre_nc = prepare_data(np.load(file=thre_nc), len_recent_time).to(device)

neigh_road_index = get_neigh_index(dataset + '/' + 'road_ad.txt').to(device)
neigh_record_index = get_neigh_index(dataset + '/' + 'record_ad.txt').to(device)
neigh_poi_index = get_neigh_index(dataset + '/' + 'poi_ad.txt').to(device)
print(all_data.shape, thre_nc.shape, label.shape)
train_x = all_data[:int(len(all_data) * 0.6)]
train_y = label[:int(len(label) * 0.6)]
train_thre_nc = thre_nc[:int(len(thre_nc) * 0.6)]
val_x = all_data[int(len(all_data) * 0.6):int(len(all_data) * 0.8)]
val_y = label[int(len(label) * 0.6):int(len(label) * 0.8)]
val_thre_nc = thre_nc[int(len(thre_nc) * 0.6):int(len(thre_nc) * 0.8)]
test_x = all_data[int(len(all_data) * 0.8):]
test_y = label[int(len(label) * 0.8):]
test_thre_nc = thre_nc[int(len(thre_nc) * 0.8):]

model = SNIPER(dr, len_recent_time, number_sp, number_region, neigh_poi_index, neigh_road_index,
               neigh_record_index).to(device)
learning_rate = params.learning_rate
trainer = optim.Adam(model.parameters(), lr=learning_rate)


def train_one_step(x, y):
    y_pred, y_dy, dy_diff = model(x)
    l, focal_loss, dy_loss = loss_function(y_pred, y, dy_diff)
    trainer.zero_grad()
    l.backward()
    trainer.step()
    training_loss = l.cpu().item()
    print(training_loss)
    return y_dy.detach()


batch_size = params.batch_size
batch_train = math.ceil((len(train_x)) / batch_size)
batch_val = math.ceil((len(val_x)) / batch_size)
training_epoch = params.training_epoch
start = time.time()
for epoch in range(0, training_epoch):
    model.train()
    print('epoch:', epoch)
    i = 0
    train_x_batch = [train_x[i * batch_size:(i + 1) * batch_size],
                     train_thre_nc[i * batch_size:(i + 1) * batch_size],
                     torch.zeros((len_recent_time, number_region, 2 * dr)).to(device)]
    train_y_batch = train_y[i * batch_size:(i + 1) * batch_size]
    y_dynamic = train_one_step(train_x_batch, train_y_batch)

    for i in range(1, batch_train):
        train_x_batch = [train_x[i * batch_size:(i + 1) * batch_size],
                         train_thre_nc[i * batch_size:(i + 1) * batch_size], y_dynamic]
        train_y_batch = train_y[i * batch_size:(i + 1) * batch_size]
        print('epoch:', epoch)
        y_dynamic = train_one_step(train_x_batch, train_y_batch)
        # earlystop...
# ...
