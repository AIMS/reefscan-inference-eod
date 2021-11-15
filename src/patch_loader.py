from PIL import Image
from keras.utils.data_utils import Sequence
import numpy as np
import keras
import time


def get_patch(dict_item):
    # filename = dict_item["patch_file"]
    filename = "c:\\patches\\" + dict_item["FILE_NAME"] + "-point3.jpg"
    patch = Image.open(filename)
    patch = keras.preprocessing.image.img_to_array(patch)
    # patch = np.expand_dims(patch, axis=0)
    return patch


class PatchLoader(Sequence):

    def __init__(self, patches, batch_size):
        self.patches = patches
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.patches) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch = self.patches[idx * self.batch_size:(idx + 1) * self.batch_size]

        indexes = []
        for index in range(idx * self.batch_size, (idx + 1) * self.batch_size):
            indexes.append(index)

        patches = [get_patch(dict_item)
            for dict_item in batch]

        array = np.array(patches)
        return array


