import os
import sys
import json

CURRENT_DIR = "./"

result = {}

for curDir, dirs, files in os.walk(CURRENT_DIR):
    if curDir.endswith("augmented") and os.path.isfile(curDir + "/" + "label.json"):
        f = open(curDir + "/" + "label.json")
        data = json.load(f)
        for x in data:
            filename = curDir + "/" + data[x]["filename"]
            data[x]["filename"] = filename[2:]
            #print(filename)
        result.update(data)
        
with open(CURRENT_DIR + "via_region_data.json", 'w') as f:
    json.dump(result, f)