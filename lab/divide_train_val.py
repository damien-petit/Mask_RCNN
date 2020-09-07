import os
import sys
import json
import random

train_file = "./train/train_data.json"
val_file = "./train/val_data.json"

f = open(train_file)
data = json.load(f)
key = list(data.keys())
random.shuffle(key)

train_length = int(len(data) * 0.8) + 1
train_key = key[:train_length]
val_key = key[train_length:]

train_data = {}
for key in train_key:
    train_data[key] = data[key]

val_data = {}
for key in val_key:
    val_data[key] = data[key]

f.close()

with open(train_file, "w") as f:
    json.dump(train_data, f)

with open(val_file, "w") as f:
    json.dump(val_data, f)
