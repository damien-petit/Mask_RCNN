import os
import sys
import time
import numpy as np
import argparse
import matplotlib.pyplot as plt
import skimage.io

# Import Mask RCNN
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import random_colors, apply_mask
import mrcnn.model as modellib
from mrcnn.model import log
import cv2
import lab_config

config = lab_config.InferenceConfig()

# parser setting
parser = argparse.ArgumentParser(description='Test model whether it can detect any images.')
parser.add_argument('--weights', required=True, metavar="/path/to/weights.h5", help="Path to weights .h5 file or 'coco'")
parser.add_argument('--image', required=True, metavar="/path/to/data/image.png", help='image to detect or visualize')
parser.add_argument('--depth', required=False, metavar="/path/to/data/depth.npy", help='depth to detect')
args = parser.parse_args()

if args.depth == None and 'depth' in lab_config.INPUT_DATA_TYPE:
    raise Exception("Need --depth option")
weight_filename = args.weights
image_filename = args.image
if args.depth != None:
    depth_filename = args.depth

# input data setting
texture = skimage.io.imread(image_filename)
vis_image = np.copy(texture)
if 'gray' in lab_config.INPUT_DATA_TYPE:
    texture = skimage.color.rgb2gray(texture)
    texture = texture.reshape([texture.shape[0], texture.shape[1], 1])
texture = lab_config.NORMALIZE_TEXTURE(texture)
if 'depth' in lab_config.INPUT_DATA_TYPE:
    depth = np.load(depth_filename)
    depth = depth.reshape([depth.shape[0], depth.shape[1], 1])
    depth = lab_config.NORMALIZE_DEPTH(depth)

if lab_config.INPUT_DATA_TYPE == 'rgb':
    input_data =  texture
elif lab_config.INPUT_DATA_TYPE == 'gray':
    input_data = texture
elif lab_config.INPUT_DATA_TYPE == 'depth':
    input_data =  depth
elif lab_config.INPUT_DATA_TYPE == 'rgb-depth':
    input_data = np.append(texture, depth, axis=2)
elif lab_config.INPUT_DATA_TYPE == 'gray-depth':
    input_data = np.append(texture, depth, axis=2)

# lead model
model = modellib.MaskRCNN(mode="inference", model_dir="./", config=config)
model.load_weights(weight_filename, by_name=True)

# detect image
results = model.detect([input_data], verbose=1)
result = results[0]

# visualize detected result
detected_instances = len(result['class_ids'])
colors = random_colors(detected_instances)

for i in range(detected_instances):
    color = colors[i]
    mask = result['masks'][...,i]
    vis_image = apply_mask(vis_image, mask ,color)

    score = result['scores'][i]
    class_id = result['class_ids'][i]
    class_name = lab_config.CLASS_NAMES[class_id]
    y1, x1, y2, x2 = result['rois'][i]

    class_caption = class_name + "(" +  str(round(score, 3)) + ")"
    cv2.putText(vis_image, class_caption, (x1 + int((x2 - x1)/3), y1 + int((y2 - y1)/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

plt.imshow(vis_image)
plt.show()
