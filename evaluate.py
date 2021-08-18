from typing import List
from dataset import *
from skimage import io
import numpy as np
from models import *
from pathlib import Path
from glob import glob
import re
import random
import pickle
from datetime import datetime
import SimpleITK as sitk
import tensorflow as tf
from skimage.transform import resize
from dataset import resize_image, preprocess_gt, preprocess_im
import os
from skimage.filters import threshold_yen, threshold_otsu
from sklearn.metrics import matthews_corrcoef, confusion_matrix
from dataset import io_load_image


def postprocess(array: np.ndarray):
    array /= 255.0
    if np.max(array) == 0:
        return array
    th = 0
    try:
        th = threshold_otsu(array)
    except RuntimeWarning:
        print(np.max(array))
    return array > th


def sklearn_dice(a: np.ndarray, b: np.ndarray):
    mat: np.ndarray = confusion_matrix(a.flatten(), b.flatten())
    _, fp, fn, tp = [int(x) for x in mat.ravel()]
    return (2*tp)/(fp + fn + 2*tp)


def sklearn_matthews_corrcoef(a: np.ndarray, b: np.ndarray):
    mat: np.ndarray = confusion_matrix(
        a.flatten(), b.flatten()).astype(np.int64)
    tn, fp, fn, tp = [int(x) for x in mat.ravel()]
    nominator = (tp*tn - fp*fn)
    denominator = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return nominator / denominator

# - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - -


def calculate_ircad_results() -> list:
    # Image path
    images_path = Path('./data/snorkel_vessels/')
    data_path = Path('./data/ircad_iso/')
    vessel_name = 'vesselsIso.nii'
    global_mask_name = 'liverMaskIso.nii'
    masks_name = ['liverMaskIso.nii', 'dilatedVesselsMaskIso.nii',
                  'bifurcationsMaskIso.nii']

    result_array = []

    for image_path in glob(str(images_path / '*')):
        path = Path(image_path)
        name = data_path / path.name[:-4]

        image_array, _ = io_load_image(image_path)
        vessel_array, _ = io_load_image(name / vessel_name)
        global_array, _ = io_load_image(name / global_mask_name)
        image_array *= global_array
        vessel_array *= global_array

        result_dict = dict(
            zip(masks_name, [{'mcc': 0, 'dice': 0} for i in range(len(masks_name))]))

        for mask_name in masks_name:
            mask_array, _ = io_load_image(str(name / mask_name))
            ia, va = (image_array *
                      mask_array) > 0, (vessel_array * mask_array) > 0
            result_dict[mask_name]['dice'] = sklearn_dice(ia, va)
            result_dict[mask_name]['mcc'] = sklearn_matthews_corrcoef(ia, va)

        print('*' * 50)
        print(name)
        for key in result_dict.keys():
            print(key, result_dict[key])

        result_array.append(result_dict)

    return result_array


def global_average_results(results: List[dict]):
    avg = dict()
    for key in results[0].keys():
        avg[key] = {}
        for subkey in results[0][key]:
            avg[key][subkey] = 0

        for element in results:
            for subkey in element[key].keys():
                avg[key][subkey] += element[key][subkey] / len(results)
    print(avg)
    return avg


if __name__ == '__main__':
    # result = calculate_ircad_results()
    # with open('./temp/ircad_res.pickle', 'wb') as file:
    #     pickle.dump(result, file)
    names = [Path(name).name[:-4]
             for name in glob(str(Path('./data/snorkel_vessels/') / '*'))]
    print(names)

    result: list = None
    with open('./temp/ircad_res.pickle', 'rb') as file:
        result = pickle.load(file)

    global_average_results(result)
