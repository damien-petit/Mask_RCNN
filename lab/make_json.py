import os
import sys
import json

CURRENT_DIR = "./"

args = sys.argv
if len(args) != 2:
    print("Need option 'train' or 'val'")
    sys.exit()
if args[1] != "train" and args[1] != "val":
    print("Accetable option is 'train' or 'val'")
    sys.exit()

CURRENT_DIR += args[1] + "/"


result = {}

for curDir, dirs, files in os.walk(CURRENT_DIR):
    if curDir.endswith("augmented") and os.path.isfile(curDir + "/" + "label_augment.json"):
        f = open(curDir + "/" + "label_augment.json")
        data = json.load(f)
        for x in data:
            filename = curDir + "/" + data[x]["filename"]
            data[x]["filename"] = filename[len(args[1]) + 1 + 2:]
            #print(filename)
        result.update(data)
        
with open(CURRENT_DIR + args[1] + "_data.json", 'w') as f:
    json.dump(result, f)