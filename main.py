import glob
import pathlib
import os
import numpy as np

from image_utils import bbox2_3D, io_load_image, io_save_image

if __name__ == '__main__':
    for i in range(1, 21):
        name = f'3Dircadb1.{i}.nii'
        v = f'./data/vessels/{name}'
        s = f'./data/snorkel_vessels/{name}'
        vv, _ = io_load_image(v)
        ss, im = io_load_image(s)

        merged = np.array(vv, dtype='int') + ss
        merged[merged > 0] = 255.0
        merged = np.array(merged, dtype='uint8')
        io_save_image(f'./data/merged_vessels/{name}', merged, im)

    pass
