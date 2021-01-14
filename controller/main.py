import os
import sys

import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog
from matplotlib import pyplot as plt
from Dicom.Get_Contour_Brain_CT.file.ChangeToTFRecord import TfRecordConversion
from Dicom.Get_Contour_Brain_CT.file.file_in_out import FileInOut
from Dicom.Get_Contour_Brain_CT.image_utils.preprocessor import Image_PreProcessor
from Dicom.Get_Contour_Brain_CT.controller.setting_controller import SettingController
from Dicom.Get_Contour_Brain_CT.ui.ui import Ui_Dialog


class Controller(object):
    def __init__(self, file_extension=".dcm"):
        self.file_paths = []
        self.dicom_files = []
        self.images_hu_pixels = []
        self.file_io = FileInOut(file_extension)
        self.image_preprocessor = Image_PreProcessor()
        self.setting_value = {
            "hu_boundary_value": -200,
            "brain_init_position": (250,145)
        }
        self.default_setting_value = self.setting_value

    def set_setting_value(self, setting_value):
        self.setting_value = setting_value
        #self.images_hu_pixels = self.image_preprocessor.get_pixels_hu(self.dicom_files)

    def start(self):
        app = QtWidgets.QApplication(sys.argv)
        self.ui = Ui_Dialog(self)
        self.ui.show()
        sys.exit(app.exec_())

    def load_files(self):
        selected_dir = QFileDialog.getExistingDirectory(None, caption='Choose Directory', directory=os.getcwd())
        self.file_paths = self.file_io.search(selected_dir)
        # self.file_paths = self.file_io.search("/Users/joonhyoungjeon/Downloads/20201221/S168/CT")

        self.ui.listView.clear()
        for filepath in self.file_paths:
            self.ui.listView.addItem(filepath[1])
        self.dicom_files = self.image_preprocessor.load_scan(self.file_paths)
        #self.images_hu_pixels = self.image_preprocessor.get_pixels_hu(self.dicom_files)

    def add_files(self):
        selected_dir = QFileDialog.getExistingDirectory(None, caption='Choose Directory', directory=os.getcwd())
        searched_files = self.file_io.search(selected_dir)
        self.file_paths.extend(searched_files)
        for filepath in searched_files:
            self.ui.listView.addItem(filepath[1])
            self.dicom_files.extend(self.image_preprocessor.load_scan([filepath]))

        #self.images_hu_pixels = self.image_preprocessor.get_pixels_hu(self.dicom_files)

    def delete_files(self):
        for index, selected_index in enumerate(self.ui.listView.selectedIndexes()):
            self.file_paths.pop(selected_index.row() - index)
            self.dicom_files.pop(selected_index.row() - index)
            self.ui.listView.takeItem(selected_index.row() - index)

        #self.images_hu_pixels = self.image_preprocessor.get_pixels_hu(self.dicom_files)

    def itemClicked(self):
        selected_index = self.ui.listView.currentRow()
        images_hu_pixels = self.image_preprocessor.get_pixels_hu([self.dicom_files[selected_index]])
        brain_crop_images = self.dicom_preprocess(images_hu_pixels[0], images_hu_pixels[0])
        plt.figure(num=self.ui.listView.currentItem().text(), figsize=(5, 3), dpi=200)
        plt.subplot(1, 2, 1)
        plt.axis('off')
        plt.tight_layout()
        plt.imshow(images_hu_pixels[0], cmap='bone')
        plt.title("Original Image")
        plt.subplot(1, 2, 2)
        plt.axis('off')
        plt.tight_layout()
        plt.imshow(brain_crop_images, cmap='bone')
        plt.title("Crop Image")
        plt.show()

    def dicom_preprocess(self,paint_crop_image, cropping_image):
        binary_image = self.image_preprocessor.get_binary_image_with_hu_value(cropping_image, hu_boundary_value=self.setting_value.get("hu_boundary_value"))

        _, contours, hierarchy = self.image_preprocessor.find_dicom_Countour(binary_image)
        brain_contour = self.image_preprocessor.find_brain_contour(contours=contours, hierarchy=hierarchy,
                                                              init_position=self.setting_value.get("brain_init_position"))
        output = self.image_preprocessor.extract_image_with_contour(image=paint_crop_image, brain_contour=brain_contour)
        return np.array(output)

    def openSetting(self):
        selected_index = self.ui.listView.currentRow()
        if selected_index == -1:
            self.ui.show_popup_ok("Warning", "한개의 이미지를 선택 후 동작해주세요")
            return
        setting_controller = SettingController(self, self.file_paths[selected_index])
        setting_controller.start()

    def file_save(self):
        croped_image = [self.dicom_preprocess(file.pixel_array, image) for file, image in zip(self.dicom_files, self.image_preprocessor.get_pixels_hu(self.dicom_files))]
        for file, brain in zip(self.dicom_files, croped_image):
            file.PixelData = brain.astype(np.uint16).tobytes()
        self.file_io.save(self.file_paths, self.dicom_files)

    def change_dicom_to_tfRecord(self):
        fileChanger = TfRecordConversion()
        fileChanger.converse(self.file_paths)

if __name__ == '__main__':
    controller = Controller(".dcm")
    controller.start()
