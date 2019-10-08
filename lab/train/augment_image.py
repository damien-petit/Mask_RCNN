import os
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import imgaug as ia
from imgaug import augmenters as iaa
import imageio
import time

ROOT_DIR = os.path.abspath("./")
DATA_DIR = os.path.join(ROOT_DIR, "data/")

iteration = 100

for curDir, dirs, files in os.walk(DATA_DIR):
    if curDir.endswith("augmented"):
        for file in files:
            if file.endswith(".png"):
                os.remove(curDir + "/" + file)

t1 = time.time()
total = 0

for curDir, dirs, files in os.walk(DATA_DIR):
    if not curDir.endswith("augmented"):
        count = 1
        print("Augmenting Director: ", curDir)
        for file in files:
            if file.endswith(".png"):
                for i in range(iteration):
                    img = imageio.imread(curDir + "/" + file)
                    seq = iaa.Sequential([
                        iaa.CropToFixedSize(width=1080, height=1080, position="center"),
                        iaa.Sometimes(0.8, iaa.Affine(rotate=(-180, 180))),
                        iaa.LogContrast(gain=(0.6, 1.4)),
                        # iaa.Sometimes(0.5, iaa.MultiplyHueAndSaturation(mul_saturation=(0.5, 1.5))),
                        iaa.Sometimes(0.5, iaa.Affine(shear=(-3, 3))),
                        # iaa.Sometimes(0.5, iaa.PiecewiseAffine(scale=(0.01, 0.02))),
                        iaa.Sometimes(0.3, iaa.GaussianBlur(sigma=(0, 1.0)))
                    ])
                    aug_img = seq.augment_image(img)
                    imageio.imwrite(curDir + "/augmented/" + str(count) + ".png", aug_img)
                    count+=1
                    total+=1

t2 = time.time()
elapsed_time = t2 - t1

print("Total of data: ", total)
print("Total of time: ", elapsed_time)

print("End of augment image")