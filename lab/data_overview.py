import os
import sys
import json

## Overview train data
print("================  Training Data Overview (Source) ===============")
print("")
class_names = {}
image_count = 0
for curDir, dirs, files in os.walk("./train"):
    if os.path.isfile(curDir + "/" + "label.json"):
        data = json.load(open(curDir + "/" + "label.json"))
        data = list(data.values())
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

print("")
print("================  Training Data Overview (Source + Augmented) ===============")
print("")
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

print("")

## Overview validation data
print("================  Validation Data Overview (Source) ===============")
print("")
class_names = {}
image_count = 0
for curDir, dirs, files in os.walk("./val"):
    if os.path.isfile(curDir + "/" + "label.json"):
        data = json.load(open(curDir + "/" + "label.json"))
        data = list(data.values())
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

print("")
print("================  Validation Data Overview (Total)  ===============")
print("")
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

