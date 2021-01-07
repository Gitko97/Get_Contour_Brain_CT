import pydicom
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from pydicom.pixel_data_handlers import util


class Image_PreProcessor(object):

    def load_scan(self, file_path_list):
        slices = [pydicom.read_file(os.path.join(s[0], s[1])) for s in file_path_list]
        # slices.sort(key=lambda x: int(x.InstanceNumber))

        return slices

    def get_pixels_hu(self, dicom_files):
        # pydicom's apply_modality_lut method has similar function with this method
        images = np.stack([s.pixel_array for s in dicom_files])
        print(images.min(), images.max())
        hu_pixels = []
        for file, image in zip(dicom_files, images):
            rescaled_arr = util.apply_modality_lut(image, file)
            # windowed_rescaled_arr = util.apply_voi_lut(rescaled_arr, file, index=0)
            hu_pixels.append(rescaled_arr)
        return np.array(hu_pixels, dtype=np.int16)

    def get_binary_image_with_hu_value(self, hu_image, hu_boundary_value=-70):
        binary_image = np.where(hu_image < hu_boundary_value, 0, 1)

        return np.array(binary_image, dtype=np.int16)

    def window_image(self, image, window_center, window_width):
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        window_image = image.copy()
        window_image[window_image < img_min] = img_min
        window_image[window_image > img_max] = img_max

        return window_image

    def normalize(self, images, min_bound=-1000, max_bound=2000, pixel_mean=None):
        images = (images - min_bound) / (max_bound - min_bound)
        images[images > 1] = 1.
        images[images < 0] = 0.
        if pixel_mean == None:
            pixel_mean = images.mean()
        print("Mean : {}, Min : {}, Max : {}".format(pixel_mean, images.min(), images.max()))
        image = images - pixel_mean
        return images.mean(), np.array(image, dtype=np.float64)

    def find_dicom_Countour(self, binary_image):
        if binary_image.dtype != np.uint8:
            binary_image = binary_image.astype(np.uint8)
        # plt.figure(None, figsize=(5, 3), dpi=200)
        # plt.subplot(1, 2, 1)
        # plt.axis('off')
        # plt.tight_layout()
        # plt.imshow(binary_image, cmap='bone')
        # plt.title("Binary Image")
        image, contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # plt.subplot(1, 2, 2)
        # plt.axis('off')
        # plt.tight_layout()
        # con = np.zeros([512,512,3])
        # cv2.drawContours(con, contours, -1, (0, 0, 255), 1)
        # plt.imshow(con, cmap='bone')
        # plt.title("Contour")
        # plt.show()
        return image, contours, hierarchy

    def get_binary_image_with_adaptiveThreshold(self, images, contour_boundary_value=100, block_size=11, c=2):
        if images.dtype != np.uint8:
            images = images.astype(np.uint8)
        mask = cv2.adaptiveThreshold(images, contour_boundary_value, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                     block_size, c)
        return mask

    def get_binary_image_with_OTSU(self, images, contour_boundary_value=10):
        if images.dtype != np.uint8:
            images = images.astype(np.uint8)
        thr, mask = cv2.threshold(images, contour_boundary_value, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return thr, mask

    def get_binary_image_with_Canny(self, images, min_value=10, max_value=200):
        if images.dtype != np.uint8:
            images = images.astype(np.uint8)
            images += images.min()
        mask = cv2.Canny(images, min_value, max_value)
        return mask

    def find_brain_contour(self, contours, hierarchy, init_position):
        brain_contour = []
        for index, contour in enumerate(contours):
            if cv2.pointPolygonTest(contour, init_position, measureDist=True) >= 0 and hierarchy[0][index][3] == -1:
                brain_contour.append(contour)

        return np.array(brain_contour)

    def extract_image_with_contour(self, image, brain_contour):
        mask = np.zeros_like(image)
        cv2.drawContours(mask, brain_contour, -1, 255, -1)
        out = np.zeros_like(image) + image.min()
        out[mask == 255] = image[mask == 255]
        return np.array(out, dtype=np.float64)

    def print_value_distribution(self, array, value_name="values", bins=100):
        plt.hist(array.flatten(), bins=bins, color='c')
        plt.xlabel(value_name)
        plt.ylabel("Frequency")
        plt.show()
