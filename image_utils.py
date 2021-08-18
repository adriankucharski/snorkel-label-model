import numpy as np
import numpy as np
from pathlib import Path
import pickle
import SimpleITK as sitk
from skimage.transform import resize
from typing import Tuple
from SimpleITK import Image

# ************************************************************
#                       IMAGE IO
# ************************************************************
from typing import List


def io_load_image(path) -> Tuple[np.ndarray, Image]:
    """
    Return an array and an image header readed by SimpleITK.ReadImage

    Parameters
    ----------
    path : str | Path
        Path of an image.

    Returns
    -------
    [array, image]
    """
    image: Image = sitk.ReadImage(str(Path(path)))
    array: np.ndarray = sitk.GetArrayFromImage(image)
    return (array, image)


def io_save_image(path: str, array: np.ndarray, image_header_data: sitk.Image = None):
    image = sitk.GetImageFromArray(array)
    if image_header_data is not None:
        image.CopyInformation(image_header_data)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(str(Path(path)))
    writer.Execute(image)


def io_load_pair_images(im_path, gt_path) -> Tuple[np.ndarray, np.ndarray]:
    X: np.ndarray = sitk.GetArrayFromImage(sitk.ReadImage(str(Path(im_path))))
    Y: np.ndarray = sitk.GetArrayFromImage(sitk.ReadImage(str(Path(gt_path))))
    return (X, Y)


def save_dataset(im_data, save_path: str = 'data/im_data.pickle'):
    with open(str(Path(save_path)), 'wb') as file:
        pickle.dump(im_data, file)


def load_dataset(path: str = 'data/im_data.pickle') -> tuple:
    im_data = None
    with open(str(Path(path)), 'rb') as file:
        im_data = pickle.load(file)
    return im_data


# ************************************************************
#                       PREPROCESS
# ************************************************************
def sigmoid(x):
    e: np.ndarray = np.exp(-x)
    return 1.0 / (1.0 + e)


def resize_image(data, static_size: tuple):
    new_size = list(data.shape)
    for i in range(len(static_size)):
        new_size[i] = new_size[i] if static_size[i] == None else static_size[i]

    data_resized = resize(data, new_size, anti_aliasing=False)
    return np.array(data_resized, dtype=np.float16)


def preprocess_im(im):
    im = (im - np.mean(im)) / np.std(im)
    return np.reshape(im, (-1, *im.shape, 1))


def preprocess_gt(gt, label=(0, 1)):
    depth, height, width = gt.shape
    gt = np.reshape(gt, (depth, height, width))
    new_gt = np.zeros((depth, height, width, len(label)), dtype='float16')
    for i in range(0, len(label)):
        new_gt[0:depth, 0:height, 0:width, i] = (gt == label[i])

    return np.reshape(new_gt, (-1, *new_gt.shape))


def bbox2_3D(img: np.ndarray):
    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return np.index_exp[rmin:rmax, cmin:cmax, zmin:zmax]
