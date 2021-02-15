"""
Mask R-CNN
Train on the toy dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import sys
import json
import time
import datetime
import numpy as np
import skimage.draw
import lab_config

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

config = lab_config.LabConfig()

############################################################
#  Dataset
############################################################

class LabDataset(utils.Dataset):

    def load_lab(self, dataset_dir, subset):
        """Load a subset of the Lab dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        for i, name in enumerate(class_names):
            if name == "BG": continue
            self.add_class("lab", i, name)

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        annotations = json.load(open(os.path.join(dataset_dir, subset + "_data.json")))
        annotations = list(annotations.values())

        # The VIA tool saves images in the JSON even if they don't have any annotations.
        # Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        # label = {}
        print("Loading", subset, " Image Dataset......")
        start = time.time() 
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
                names = [r['region_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']] 
                names = [r['region_attributes'] for r in a['regions']]

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read the image.
            # This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "lab",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons, names=names)

        end = time.time()
        print("Finshed Loading", subset, " Image dataset")
        print("Loading Time: ", round((end - start), 3) ,  "[s]")

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a lab dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "lab":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        class_ids = []
        for i, (p, n) in enumerate(zip(info["polygons"], info["names"])):
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1
            class_id = class_names.index(n["name"])
            class_ids.append(class_id)

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        # return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)
        return mask.astype(np.bool), np.array(class_ids, dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "lab":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,CHANNEL_COUNT] Numpy array.
        """
        # Load image and depth
        filename = self.image_info[image_id]['path']
        if 'rgb' in lab_config.INPUT_DATA_TYPE or 'gray' in lab_config.INPUT_DATA_TYPE:
            texture = skimage.io.imread(filename)
            if 'gray' in lab_config.INPUT_DATA_TYPE:
                texture = skimage.color.rgb2gray(texture)
                texture = texture.reshape([texture.shape[0], texture.shape[1], 1])
            texture = lab_config.NORMALIZE_TEXTURE(texture)

        if 'depth' in lab_config.INPUT_DATA_TYPE:
            depth = np.load(filename.split(".png")[0] + "_" + lab_config.DEPTH_FILE_END_WORD)
            depth = depth.reshape([depth.shape[0], depth.shape[1], 1])
            depth = lab_config.NORMALIZE_DEPTH(depth)

        if lab_config.INPUT_DATA_TYPE == 'rgb':
            return texture
        elif lab_config.INPUT_DATA_TYPE == 'gray':
            return texture
        elif lab_config.INPUT_DATA_TYPE == 'depth':
            return depth
        elif lab_config.INPUT_DATA_TYPE == 'rgb-depth':
            image = np.append(texture, depth, axis=2)
            return image
        elif lab_config.INPUT_DATA_TYPE == 'gray-depth':
            image = np.append(texture, depth, axis=2)
            return image


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = LabDataset()
    dataset_train.load_lab(args.dataset, "train")
    dataset_train.prepare()

    num_train_images = dataset_train.num_images
    model.config.STEPS_PER_EPOCH = int(num_train_images / (model.config.GPU_COUNT * model.config.IMAGES_PER_GPU)) + 1 

    # Validation dataset
    dataset_val = LabDataset()
    dataset_val.load_lab(args.dataset, "val")
    dataset_val.prepare()

    # Start training
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=lab_config.EPOCH,
                layers=lab_config.TRAINING_LAYER)

############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect lab data.')
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/lab/dataset/",
                        help='Directory of the lab dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "detect":
        assert args.image, "Argment --image is required for detection"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    config.display()

    # Create model
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    elif weights_path = "none":
        pass
    else:
        model.load_weights(weights_path, by_name=True)

    train(model)
