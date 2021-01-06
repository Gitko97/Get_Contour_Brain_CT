import pydicom
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt


class Image_PreProcessor(object):

    def load_scan(self, file_path_list):
        slices = [pydicom.read_file(os.path.join(s[0], s[1])) for s in file_path_list]
        # slices.sort(key=lambda x: int(x.InstanceNumber))

        return slices

    def get_pixels_hu(self, dicom_files, hu_boundary_value=-1000):
        image = np.stack([s.pixel_array for s in dicom_files])
        image = image.astype(np.int16)
        # image[image <= hu_boundary_value] = 0

        for slice_number in range(len(dicom_files)):

            intercept = dicom_files[slice_number].RescaleIntercept
            slope = dicom_files[slice_number].RescaleSlope
            if slope != 1:
                image[slice_number] = slope * image[slice_number].astype(np.float64)
                image[slice_number] = image[slice_number].astype(np.int16)

            image[slice_number] += np.int16(intercept)
        return np.array(image, dtype=np.int16)

    def normalize(self, images, min_bound=-1000, max_bound=500, pixel_mean=None):
        images = (images - min_bound) / (max_bound - min_bound)
        images[images > 1] = 1.
        images[images < 0] = 0.
        if pixel_mean == None :
            pixel_mean = images.mean()
        print("Mean : {}, Min : {}, Max : {}".format(pixel_mean, images.min(), images.max()))
        image = images - pixel_mean
        return images.mean(), np.array(image, dtype=np.float64)

    def find_dicom_Countour_OTSU(self, hu_images, contour_boundary_value=100):
        if hu_images.dtype != np.uint8:
            hu_images = hu_images.astype(np.uint8)
        thr, mask = cv2.threshold(hu_images, contour_boundary_value, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        image, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return image, contours, hierarchy

    def find_brain_contour(self, contours,hierarchy, init_position):
        brain_contour = []
        for index, contour in enumerate(contours):
            if cv2.pointPolygonTest(contour, init_position, measureDist=True) >= 0 and hierarchy[0][index][3] != -1:
                if hierarchy[0][hierarchy[0][index][3]][3] == -1:
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