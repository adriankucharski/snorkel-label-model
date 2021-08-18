from glob import glob
from pathlib import Path

from ModLabelModel import LabelModelFixed
from typing import AnyStr, Callable
import numpy as np
import SimpleITK as sitk
import snorkel
from skimage import io
from skimage.color import rgb2gray
from skimage.filters import (threshold_li, threshold_otsu, threshold_sauvola, threshold_mean,
                             threshold_yen, threshold_isodata, threshold_triangle)
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation, diamond, erosion
from snorkel.labeling import LFApplier, labeling_function
from snorkel.labeling.model import LabelModel
from scipy.ndimage.morphology import binary_dilation, binary_erosion, binary_closing
import re
from dataset import io_load_image, io_save_image
from collections import Counter, defaultdict


def calculate_dice(a: np.ndarray, b: np.ndarray):
    pred = a.flatten()
    true = b.flatten()

    pred = pred / np.max(pred)
    true = true / np.max(true)

    intersection = np.sum(pred * true) * 2.0
    dice = intersection / (np.sum(pred) + np.sum(true))
    return dice


def sorting(s): return int(re.findall(r'/d+', s)[-1])


def get_largest_connected_component(segmentation: np.ndarray, cc_elements=5):
    assert (cc_elements >= 1)
    labels: np.ndarray = label(segmentation, connectivity=2)
    if labels.max() == 0:  # assume at least 1 CC
        print("0 CC elements!")
        return np.zeros(segmentation.shape)
    if labels.max() < cc_elements:
        cc_elements = labels.max()
    print(f'Labels: {labels.max()}, Elements: {cc_elements}')

    largest_cc = np.zeros(labels.shape)
    count = np.bincount(labels.flat)[1:]
    for i in range(cc_elements):
        largest_cc += (labels == np.argmax(count) + 1)
        count[np.argmax(count)] = 0

    return largest_cc


def ignore_black_background(im):
    # return im
    im = im.flatten()
    return im[np.where(im != 0)]


def li_thresholding(im):
    im_mod = ignore_black_background(im)
    threshold = threshold_li(im_mod)
    return np.array(im > threshold, dtype=np.uint8)


def otsu_thresholding(im):
    im_mod = ignore_black_background(im)
    threshold = threshold_otsu(im_mod)
    return np.array(im > threshold, dtype=np.uint8)


def yen_thresholding(im):
    im_mod = ignore_black_background(im)
    threshold = threshold_yen(im_mod)
    return np.array(im > threshold, dtype=np.uint8)


def triangle_thresholding(im):
    im_mod = ignore_black_background(im)
    threshold = threshold_triangle(im_mod)
    return np.array(im > threshold, dtype=np.uint8)


def mean_thresholding(im):
    im_mod = ignore_black_background(im)
    threshold = threshold_mean(im_mod)
    return np.array(im > threshold, dtype=np.uint8)


def sauvola_thresholding(im):
    threshold = threshold_sauvola(im, window_size=21)
    return np.array(im > threshold, dtype=np.uint8)


def isodata_thresholding(im):
    im_mod = ignore_black_background(im)
    threshold = threshold_isodata(im_mod)
    return np.array(im > threshold, dtype='uint8')


def static_thresholding(im):
    threshold = 15
    return np.array(im > threshold, dtype=np.uint8)


class Image():
    def __init__(self, labels, shape):
        self.labels = labels
        self.shape = shape


#################################################################################
#################################################################################
#################################################################################
def run_label_model_on_batch(images_path: str, paths: list, labeling_functions=[li_thresholding, isodata_thresholding, yen_thresholding], Y_dev=None, class_balance=[0.9, 0.1]):
    size = 0
    dataset = []
    for dir_path in sorted(glob(str(Path(images_path))), key=sorting):
        ip: Path = Path(dir_path)

        image_arr, header = io_load_image(str(ip))
        image_arr = rgb2gray(image_arr)
        image_shape = image_arr.shape
        image_arr = image_arr.flatten()

        labels = []
        for path in paths:
            name = ip.name.replace('training.tif', 'vessels.png')
            pp = Path(path) / name
            arr, im = io_load_image(pp)
            for lb_func in labeling_functions:
                array = lb_func(arr.flatten())
                size += array.shape[-1]
                labels.append(array)
        dataset.append([header, image_arr, labels, ip.name, image_shape])

    array_reshape = (size // len(paths) // len(labeling_functions),
                     len(paths) * len(labeling_functions))
    print(array_reshape)

    lab: np.ndarray = np.zeros((size), dtype='float16').reshape(array_reshape)
    print(size, lab.shape)

    s = 0
    for data in dataset:
        _, _, label, _, _ = data
        t: np.ndarray = np.array(label).T
        si = t.shape[0]
        lab[s:s + si, :] = t
        s += si

    print(('*' * 10) + ' Training ' + ('*' * 10))
    LM: LabelModelFixed = LabelModelFixed(
        cardinality=2, verbose=True, device='cuda')
    batch_size = lab.shape[0] // 8
    for i in range(0, lab.shape[0], batch_size):
        print(
            f'Batch {i // batch_size} of {lab.shape[0] // batch_size}' + (' ' * 20), end='/r')

        chunk = lab[i:i + batch_size, ]
        LM._fit_modified(chunk, seed=123, log_freq=1, n_epochs=100, Y_dev=Y_dev,
                         class_balance=class_balance, continue_training=(i != 0))
    LM._eval_modified()

    path_save = './data/temp'
    print(('*' * 10) + ' Prediction ' + ('*' * 10))
    for data in dataset:
        im, _, label, fn, shape = data

        t: np.ndarray = np.array(label).T
        p: np.ndarray = LM.predict(t)
        p = p.reshape(shape)
        p[p > 0] = 255
        p = np.array(p, dtype='uint8')
        io_save_image(f'{path_save}/{fn}', p, im)
        print(f'{path_save}/{fn}')


def postprocess_2D(image: np.ndarray):
    image = binary_closing(image, np.ones((3, 3)))
    image = get_largest_connected_component(image)
    image[image > 0] = 255
    return image.astype('uint8')


def postprocess_images_2D(images_path: str, save_path: str, postprocess_func: Callable, im_extension: str = 'tif'):
    for im_path in glob(f'{images_path}/*.{im_extension}'):
        im_name = Path(im_path).name

        im, header = io_load_image(im_path)
        im = postprocess_2D(im)
        io_save_image(str(Path(save_path) / im_name), im, header)


#################################################################################
#################################################################################
#################################################################################

def voting_system():
    patient = 'maskedLiverIso.nii'
    filenames = ["ruiZhang.nii", "jerman.nii"]
    labeling_functions = [
        li_thresholding, isodata_thresholding, yen_thresholding, otsu_thresholding]
    path = 'F:/Deep Learning/Data/snorkel/*'

    for dir_path in sorted(glob(str(Path(path))), key=sorting):
        ip: Path = Path(dir_path) / patient
        im, metadata = io_load_image(str(ip))
        name = Path(dir_path).name

        voting = np.zeros(im.shape)
        for fn in filenames:
            for func in labeling_functions:
                i = Path(dir_path) / fn
                array, image = io_load_image(str(i))
                array = func(array)
                voting += array

        num_of_values = len(filenames) + len(labeling_functions)
        voting[np.where(voting < (num_of_values // 2))] = 0
        voting[np.where(voting > 0)] = 1

        save_path = f'temp/{name}.nii'
        io_save_image(save_path, voting, metadata)
        print(f'Saved {save_path}')


def labeling_model_batch():
    patient = 'maskedLiverIso.nii'
    filenames = ["ruiZhang.nii", "jerman.nii"]
    labeling_functions = [li_thresholding,
                          isodata_thresholding, yen_thresholding]
    path = 'F:/Deep Learning/Data/snorkel/*'
    path_save = 'temp/snorkel_2/'
    _, gold_label = io_load_image(
        './data/ircad_full/3Dircadb1.15/vesselsIso.nii')

    dataset = []

    size = 0
    for dir_path in sorted(glob(str(Path(path))), key=sorting):
        ip: Path = Path(dir_path) / patient

        arr, im = io_load_image(str(ip))
        shape = arr.shape
        arr = arr.flatten()

        labels = []
        for fn in filenames:
            for func in labeling_functions:
                i = Path(dir_path) / fn
                array, image = io_load_image(str(i))
                array = func(array.flatten())
                size += array.shape[-1]
                labels.append(array)

        dataset.append([im, arr, labels, ip.parts[-2], shape])
    array_reshape = (size // len(filenames) // len(labeling_functions),
                     len(filenames) * len(labeling_functions))
    print(array_reshape)

    lab: np.ndarray = np.zeros((size), dtype='float16').reshape(array_reshape)
    print(size, lab.shape)
    s = 0
    for data in dataset:
        _, _, label, _, _ = data
        t: np.ndarray = np.array(label).T
        si = t.shape[0]
        lab[s:s + si, :] = t
        s += si

    print(('*' * 10) + ' Training ' + ('*' * 10))
    LM: LabelModelFixed = LabelModelFixed(
        cardinality=2, verbose=True, device='cuda')
    batch_size = lab.shape[0] // 256
    for i in range(0, lab.shape[0], batch_size):
        print(
            f'Batch {i // batch_size} of {lab.shape[0] // batch_size}' + (' ' * 20), end='/r')

        chunk = lab[i:i + batch_size, ]
        LM._fit_modified(chunk, seed=12345, log_freq=1, n_epochs=100,
                         Y_dev=gold_label, continue_training=(i != 0))

    LM._eval_modified()

    print(('*' * 10) + ' Prediction ' + ('*' * 10))

    for data in dataset:
        im, _, label, fn, shape = data

        t: np.ndarray = np.array(label).T
        p: np.ndarray = LM.predict(t)
        p = p.reshape(shape)
        p[p > 0] = 255
        p = np.array(p, dtype='uint8')
        io_save_image(f'{path_save}/{fn}', p, im)
        print(f'{path_save}/{fn}')


def lab_mod_b():
    patient = 'maskedLiverIso.nii'
    filenames = ["ruiZhang.nii", "jerman.nii"]
    labeling_functions = [
        li_thresholding, isodata_thresholding, yen_thresholding, otsu_thresholding]
    path = 'F:/Deep Learning/Data/snorkel/*'
    path_save = 'temp/'

    save_data = []
    all_images = []
    for dir_path in sorted(glob(str(Path(path))), key=sorting):
        images = []
        flag = True
        for fn in filenames:
            image, header = io_load_image(Path(dir_path) / fn)
            images.append(image)
            if flag:
                save_data.append([image.shape, header])
                flag = False
        all_images.append(images)

    LM: LabelModelFixed = LabelModelFixed(
        cardinality=2, verbose=True, device='cuda')
    LM.run_model(all_images, labeling_functions, batch_split=128,
                 logs=True, class_balance=[0.925, 0.075])

    for i in range(len(all_images)):
        pred = LM.predict_image(all_images[i], labeling_functions, False)
        print(pred)
        name = f'{path_save}/{i+1}.nii'
        print(name)
        shape, header = save_data[i]
        pred = pred.reshape(shape)
        io_save_image(name, pred, header)
        break


def mask_vessels_with_image():
    path_source = 'F:/Deep Learning/Data/ircad_iso/*'
    path_target = 'F:/Deep Learning/Data/snorkel/*'

    for im_path in glob(path_source):
        target = Path(im_path.replace('ircad_iso', 'snorkel'))

        im_liver = Path(im_path) / 'liverMaskIso.nii'
        im_vessels = Path(im_path) / 'vesselsIso.nii'

        liv, im_liv = io_load_image(im_liver)
        ves, im_ves = io_load_image(im_vessels)

        liv = binary_erosion(liv, np.ones((3, 3, 3)))
        ves[liv == 0] = 0

        im_target = target / 'vesselsIsoMasked.nii'
        io_save_image(im_target, ves, im_ves)
        print(im_target)


def postprocess_snorkel(image: np.ndarray, mask: np.ndarray):
    image = binary_dilation(image, np.ones((3, 3, 3)))
    elem = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                     diamond(1), [[0, 0, 0], [0, 1, 0], [0, 0, 0]]])
    image = binary_erosion(image, elem)
    mask = binary_erosion(mask, np.ones((9, 9, 9)))
    image[mask == 0] = 0

    # image = get_largest_connected_component(image, 340)
    image[image > 0] = 255
    return image.astype('uint8')


def postprocess_images(images_path: str, save_path: str):
    for im_path in glob(f'{images_path}/*'):
        im_name = Path(im_path).name
        print(im_name)
        mask_path = f'F:/Deep Learning/Data/ircad_iso/{im_name.replace(".nii", "")}/liverMaskIso.nii'

        im, header = io_load_image(im_path)
        mask, _ = io_load_image(mask_path)
        post = postprocess_snorkel(im, mask)

        io_save_image(str(Path(save_path) / im_name), post, header)


def process_image():
    image_name = '3Dircadb1.15'
    image_path = f'temp/{image_name}.nii'
    mask_path = f'F:/Deep Learning/Data/ircad_iso/{image_name}/liverMaskIso.nii'
    path_save = f'temp/{image_name}_p.nii'

    im, metadata = io_load_image(image_path)
    mask, _ = io_load_image(mask_path)

    im = postprocess_snorkel(im, mask)
    io_save_image(path_save, im, metadata)


if __name__ == '__main__':
    lab_mod_b()
    exit()
    # process_image()
    images_path = 'F:/Deep Learning/Data/CT Chest Scans/Low-Resolution-Fundus/datasets/training/images/*.tif'
    gold_dev: np.ndarray = io.imread(
        'F:/Deep Learning/Data/CT Chest Scans/Low-Resolution-Fundus/datasets/training/1st_manual/21_manual1.gif', as_gray=True)
    gold_dev = gold_dev.flatten()
    if 0:
        run_label_model_on_batch(images_path, [
            './data/temp/Frangi',
            './data/temp/Sato',
            './data/temp/Meijering',
        ],
            labeling_functions=[li_thresholding,
                                triangle_thresholding, mean_thresholding],
            # Y_dev=gold_dev
            class_balance=[0.9, 0.1]
        )

    postprocess_images_2D('./data/temp', './data/temp/post', postprocess_2D)

    manual_path = 'F:/Deep Learning/Data/CT Chest Scans/Low-Resolution-Fundus/datasets/training/1st_manual'

    m = 0.0
    second = 0.0
    for i in range(21, 41, 1):
        pp = f'{manual_path}/{i}_manual1.gif'
        ff = f'./data/temp/post/{i}_training.tif'
        ee = io.imread(f'./data/temp/Frangi/{i}_vessels.png', False)
        ee = mean_thresholding(ee)

        ppp = f'F:/Deep Learning/Data/CT Chest Scans/Low-Resolution-Fundus/datasets/training/mask/{i}_training_mask.gif'
        mask: np.ndarray = io.imread(ppp, True)
        mask = erosion(mask, np.ones((13, 13)))

        pp = io.imread(pp, True)
        ff = io.imread(ff, True)
        ff[mask == 0] = 0
        dice = calculate_dice(pp, ff)
        dice2 = calculate_dice(pp, ee)
        second += dice2
        m += dice
        print(i, dice, dice2)

    print(m / 20, second / 20)

    # remove_disconnected()
    # voting_system()
    # exit()
    # labeling_model_batch()
    # postprocess_images('temp/snorkel_2/', 'temp/post_2/')

    # exit()
