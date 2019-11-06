import os
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import cv2
import json
import numpy as np
import skimage.draw
from imgaug import augmenters as iaa

ROOT_DIR = os.path.abspath("./")
AUGMENT_SIZE_PER_IMAGE = 20

args = sys.argv
if len(args) != 2:
    print("Need option 'train' or 'val'")
    sys.exit()
if args[1] != "train" and args[1] != "val":
    print("Accetable option is 'train' or 'val'")
    sys.exit()

target = args[1]
print("Augment target Directory :", target)

# Augmentation definition
seq1 = iaa.Sequential([
    iaa.Sometimes(0.8, iaa.Affine(rotate=(-180, 180))),
    iaa.Sometimes(0.8, iaa.Resize((0.5, 1.1))),
    iaa.Sometimes(0.8, iaa.GaussianBlur(sigma=(0.0, 2.0)))
])
seq2 = iaa.Sequential([
    iaa.Sometimes(0.8, iaa.LogContrast(gain=(0.6, 1.1)))
])

DATA_DIR = os.path.join(ROOT_DIR, target)
ids = 0
for curDir, dirs, files in os.walk(DATA_DIR):
    if not curDir.endswith("augmented") and os.path.isfile(curDir+ "/label.json"):
        annotations = json.load(open(curDir + "/label.json"))
        annotations = list(annotations.values())
        annotations = [a for a in annotations if a['regions']]

        print("Current Directory for Augmentation:", curDir)
        result = {}
        for a in annotations:
            polygons = [r['shape_attributes'] for r in a['regions']]
            names = [r['region_attributes'] for r in a['regions']]

            img = cv2.imread(curDir + "/" + a['filename'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width = img.shape[:2]

            mask = np.zeros([height, width, len(polygons)], dtype=np.uint8)
            mask_attr = []
            for i, (p, n) in enumerate(zip(polygons, names)):
                rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
                mask[rr, cc, i] = 1
                mask_attr.append(n["name"])

            img_and_mask = np.append(img, mask, axis=2)
            for i in range(AUGMENT_SIZE_PER_IMAGE):
                filename = a['filename'].split(".")[0] + "_" + str(i) + ".png"
                # if os.path.isfile(curDir + "/augmented/" + filename):
                #     continue

                aug = seq1.augment_image(img_and_mask)
                aug_img = aug[:, :, :3]
                aug_mask = aug[:, :, 3:]

                aug_img = seq2.augment_image(aug_img)
                cv2.imwrite(curDir + "/augmented/" + filename, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))

                # Make augmented masked label data
                result[str(ids) + "_" + filename] = {"filename": filename, "file_attributes":{}, "size":0}
                regions = []
                for j in range(mask.shape[-1]):
                    attr = {}
                    attr["region_attributes"] = {"name": mask_attr[j]}

                    tmp = aug_mask[:,:,j]
                    ret, thresh = cv2.threshold(tmp, 0.5, 1.0, cv2.THRESH_BINARY)
                    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    if (len(contours) == 0) : continue

                    contours_index = 0
                    for k in range(len(contours)):
                        if k != len(contours) - 1:
                            if len(contours[k]) < len(contours[k + 1]):
                                contours_index = k + 1

                    contours = contours[contours_index]
                    all_points_x = [int(contours[k][0][0]) for k in range(len(contours))]
                    all_points_y = [int(contours[k][0][1]) for k in range(len(contours))]
                    attr["shape_attributes"] = {"name": "polyline", "all_points_x": all_points_x, "all_points_y": all_points_y}
                    regions.append(attr)

                result[str(ids) + "_" + filename]["regions"] = regions

        ids += 1

        with open(curDir + "/augmented/label_augment.json", "w") as f:
            json.dump(result, f)

print("Finished")
