import os
import pydicom
import re
from tqdm import tqdm

class FileInOut(object):

    def __init__(self, file_extension):
        self.file_extension = file_extension
        self.ct_regex = re.compile(r"S.*[CT | GD].*")

    def search(self, root_directory):
        try:
            file_list = []
            for (path, dir, files) in os.walk(root_directory):
                print(dir)
                for filename in files:
                    ext = os.path.splitext(filename)[-1]
                    if ext == self.file_extension and self.ct_regex.search(filename):
                        file_list.append([path, filename])
            return file_list
        except PermissionError:
            pass

    def save(self, file_paths, dicom_files):
        for path, file in zip(file_paths, dicom_files):
            new_file_path = path[0] + "_Croped"
            if not os.path.exists(new_file_path):
                os.makedirs(new_file_path)
            pydicom.dcmwrite(os.path.join(new_file_path, "Crop_" + path[1]), file)

    def dicom_to_tfRecord(self, root_directory):
        
        return 0