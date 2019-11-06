import os
import sys
import json

## Overview train data
print("================  Training Data Overview  ===============")
print()
target_file = "./train/train_data.json"

data = json.load(open(target_file))
data = list(data.values())
class_names = {}
image_count = 0

for d in data:
    regions = d["regions"]
    if regions != []:
        image_count += 1
        for r in regions:
            name = r["region_attributes"]["name"]
            if class_names.get(name): class_names[name] += 1
            else: class_names[name] = 1

print("The number of Train Image: ", image_count)
print("The number of Train Label: ", sum(class_names.values()))
for k, v in sorted(class_names.items()):
    print("    '" + k + "' label: ", v)

print()

## Overview validation data
print("================  Validation Data Overview  ===============")
print()
target_file = "./val/val_data.json"

data = json.load(open(target_file))
data = list(data.values())
class_names = {}
image_count = 0

for d in data:
    regions = d["regions"]
    if regions != []:
        image_count += 1
        for r in regions:
            name = r["region_attributes"]["name"]
            if class_names.get(name): class_names[name] += 1
            else: class_names[name] = 1

print("The number of Validation Image: ", image_count)
print("The number of Validation Label: ", sum(class_names.values()))
for k, v in sorted(class_names.items()):
    print("    '" + k + "' label: ", v)

