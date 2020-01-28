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

SUBSET_DIR = CURRENT_DIR + args[1] + "/"
TARGET_FILE = SUBSET_DIR + args[1] + "_data.json"
result = {}
if os.path.isfile(TARGET_FILE):
    f = open(TARGET_FILE)
    data = json.load(f)
    for x in data:
        for y in data[x]["regions"]:
            y["region_attributes"]["name"] = "box"
    result.update(data)

    with open(SUBSET_DIR + args[1] + "_data.json", 'w') as f:
        json.dump(result, f)
