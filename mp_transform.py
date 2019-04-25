import cv2
import numpy as np
import os
from argparse import ArgumentParser
from multiprocessing import Pool

ROBOTCAR_ALPHA = 0.4642


def transform(im):
    # Assumes image is RGB-ordered
    image = cv2.imread(read_dir + "/" + im, cv2.IMREAD_UNCHANGED)
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    ii_image = 0.5 + np.log(g) -\
        ROBOTCAR_ALPHA*np.log(b) - \
        (1-ROBOTCAR_ALPHA)*np.log(r)

    # Lastly, convert from float to uint8 space
    max_ii = np.max(ii_image)
    min_ii = np.min(ii_image)
    uii = np.uint8((ii_image - min_ii) * 256 / (max_ii - min_ii))
    ii_name = write_dir + "/" + im
    cv2.imwrite(ii_name, uii)
    return ii_image


def transform_loop(directory):
    image_names = os.listdir(directory)

    # Spawn 4 worker processes to transform in parallel
    p = Pool(4)
    p.map(transform, image_names)


if __name__ == '__main__':
    parser = ArgumentParser(
        description=
        'Transform images in a directory into lighting invariant color space')
    parser.add_argument('--read-dir', action="store", type=str, required=True)
    parser.add_argument('--write-dir', action="store", type=str, required=True)
    args = parser.parse_args()

    global read_dir, write_dir
    read_dir = args.read_dir
    write_dir = args.write_dir
    transform_loop(read_dir)
