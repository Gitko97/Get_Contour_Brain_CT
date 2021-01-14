import os
from typing import List
import tensorflow as tf
import numpy as np
import pandas as pd
import pydicom


class TfRecordConversion:

    @staticmethod
    def _randomly_split_train_test():
        pass

    @staticmethod
    def _make_csv(data: np.ndarray, mode: str, base_save_path: str):
        df = pd.DataFrame([os.path.join(i[0], i[1]) for i in data],
                          columns=['combined_png_path'])

        df.to_csv(f"{mode}.csv")
        pass

    def _make_features(self, combined_file_name, combined_encoded_image):
        return tf.train.Features(
            feature={
                'image/file_name': self._bytes_feature(combined_file_name),
                'image/encoded_image': self._bytes_feature(combined_encoded_image),
            }
        )

    def _make_tfrecords(self, data_paths: np.ndarray, mode: str):
        output_records_name = f"{mode}.tfrecords"
        writer = tf.compat.v1.python_io.TFRecordWriter(output_records_name)
        print(f"make {mode} tfrecords start...")
        with tf.compat.v1.Session() as sess:
            for combined_data_path in data_paths:
                file_name = os.path.basename(combined_data_path[1])
                image = pydicom.read_file(os.path.join(combined_data_path[0], combined_data_path[1])).PixelData
                example = tf.train.Example(features=self._make_features(file_name.encode(),
                                                                        image))
                writer.write(example.SerializeToString())

        writer.close()
        print(f"complete make {mode} tfrecords")

    def converse(self, dicom_paths: List):
        total_data_len = len(dicom_paths)
        print(f"total_data_len = {total_data_len}")
        train_idx = np.random.choice(total_data_len, int(total_data_len * 0.99), replace=False)
        test_idx = np.setdiff1d(range(total_data_len), train_idx)

        dicom_paths = np.array(dicom_paths)
        train_data: np.ndarray = dicom_paths[train_idx]
        test_data: np.ndarray = dicom_paths[test_idx]

        print("total_data size = ", len(dicom_paths))
        self._make_csv(train_data, 'train', "")
        self._make_tfrecords(train_data, 'train')

        self._make_csv(test_data, 'test', "")
        self._make_tfrecords(test_data, 'test')

    def _bytes_feature(self, value: str) -> bytearray:
        """
        :param value = string or byte
        :return byte list
        """
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
