import numpy as np
import cv2
import glob
import os

image_path = "./outputs/npy_files"
output_directory = "./outputs/depth_map/"
paths = glob.glob(os.path.join(image_path, '*.npy'))
for idx, image_p in enumerate(paths):
    print(image_p)
    input_image = np.load(image_p)
    # input_image = pil.open(image_p)
    print(input_image.shape)
    # input_image.squeeze()
    output_name = os.path.splitext(os.path.basename(image_p))[0]
    cv2.imwrite(output_directory+"{}.bmp".format(output_name), input_image.squeeze()*255)

