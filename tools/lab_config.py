from mrcnn.config import Config
import numpy as np

def normalize_texture(texture):
    texture = texture.astype(np.float32)
    texture /= 255.0

    return texture

def normalize_depth(depth):
    min_depth = 0.0
    max_depth = 400.0
    depth = (depth - min_depth)  / (max_depth - min_depth)
    depth[np.where(depth < 0.0)] = 0.0
    depth[np.where(depth > 1.0)] = 0.0

    return depth

NORMALIZE_TEXTURE = normalize_texture
NORMALIZE_DEPTH = normalize_depth

# heads: The RPN, classifier and mask heads of the network
# all: All the layers
# 3+: Train Resnet stage 3 and up
# 4+: Train Resnet stage 4 and up
# 5+: Train Resnet stage 5 and up
TRAINING_LAYER = "all"

# rgb, gray, depth, rgb-depth or gray-depth
# You can add more data type
INPUT_DATA_TYPE = "depth"

# depth file end name
# ex
# 0000_depth.npy
# -> 'depth.npy'
DEPTH_FILE_END_WORD = "depth_from_bin.npy"
    
EPOCH = 100

CLASS_NAMES = ['BG', 'obj']
    
############################################################
#  Configurations
############################################################
class LabConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "lab"

    # Number of images to train with on each GPU.
    # A 12GB GPU can typically handle 2 images of 1024x1024px.
    # Adjust based on your GPU memory and image sizes.
    # Use the highest number that your GPU can handle for best performance.
    IMAGES_PER_GPU = 2

    # NUMBER OF GPUs to use. When using only a CPU, this needs to be set to 1.
    # Batch size = IMAGES_PER_GPU * GPU_COUNT
    GPU_COUNT = 4

    # Number of classes (including background)
    NUM_CLASSES = len(CLASS_NAMES)

    IMAGE_CHANNEL_COUNT = int('rgb' in INPUT_DATA_TYPE) * 3 + int('gray' in INPUT_DATA_TYPE) * 1 + int('depth' in INPUT_DATA_TYPE) * 1
    
    MEAN_PIXEL = np.zeros(IMAGE_CHANNEL_COUNT)
    
class InferenceConfig(LabConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.5
    DETECTION_NMS_THRESHOLD = 0.3

