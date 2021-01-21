# -*- coding: utf-8 -*-
import numpy as np
from file.file_in_out import FileInOut
from image_utils.preprocessor import Image_PreProcessor
from imgaug import augmenters as iaa
from tqdm import tqdm


class LoadDataSet:
    def __init__(self, root_directory, input_shape=(512, 512, 1)):
        fileIO = FileInOut('.dcm')
        self.input_shape = input_shape
        self.ct_paths = fileIO.find_ct_files(root_directory)
        self.mr_paths = fileIO.find_mr_files(root_directory)
        if np.shape(self.ct_paths)[0] > np.shape(self.mr_paths)[0]:
            self.ct_paths = self.ct_paths[0: np.shape(self.mr_paths)[0]]
        else:
            self.mr_paths = self.mr_paths[0: np.shape(self.ct_paths)[0]]
        self.image_preprocessor = Image_PreProcessor()
        print(np.shape(self.mr_paths), "/", np.shape(self.ct_paths))

    def change_to_pixel_array(self):
        print("MR/CT Dicoms start loading!")
        changed_ct = self.image_preprocessor.load_scan(self.ct_paths)
        changed_mr = self.image_preprocessor.load_scan(self.mr_paths)
        print("MR/CT Dicoms loaded!")
        print("MR/CT images start changing To HU pixels")
        changed_ct = self.image_preprocessor.get_pixels_hu(changed_ct, input_shape=self.input_shape[0:2])
        changed_mr = self.image_preprocessor.get_pixels_hu(changed_mr, input_shape=self.input_shape[0:2])
        print("MR/CT images Changed To HU pixels")

        print("MR/CT images start normalizing")
        _, changed_ct = self.image_preprocessor.normalize(changed_ct,min_bound=-1000, max_bound=4000, pixel_mean=0.25)
        _, changed_mr = self.image_preprocessor.normalize(changed_mr,min_bound=-10, max_bound=2000,pixel_mean=0.25)

        # augumentation_ct = self.data_augumentation(changed_ct)
        # augumentation_mr = self.data_augumentation(changed_mr)
        #
        # ct_data = np.append(changed_ct, augumentation_ct, axis=0)
        # mr_data = np.append(changed_mr, augumentation_mr, axis=0)
        return changed_ct, changed_mr

    def data_augumentation(self, data):
        seq = iaa.Sequential([
            iaa.Fliplr(0.5),
            # iaa.Flipud(0.5),
            iaa.LinearContrast((0.75, 1.5)),
            iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
            iaa.Multiply((0.8, 1.2), per_channel=0.2)
        ],
            random_order=True)
        data = seq.augment_images(data)
        # seq.show_grid(data[0], cols=8, rows=8)
        return data
