from Dicom.Get_Contour_Brain_CT.model.generator import Generator
from Dicom.Get_Contour_Brain_CT.model.discriminator import Discriminator
import numpy as np


class LearningSequence:

    def __init__(self, input_shape):
        self.generator = Generator()
        self.generator.buildModel(input_shape)
        self.discriminator1 = Discriminator(name="one")
        self.discriminator1.buildModel(input_shape)
        self.discriminator2 = Discriminator(name="one", resue_layer=self.discriminator1.reuse_layer)
        self.discriminator2.buildModel(input_shape)

    def __call__(self):
        return self.generator.trainModel(np.ones((1, 512, 512, 1)))


if __name__ == '__main__':
    Learning_Sequence = LearningSequence((512, 512, 1))
    # print(Learning_Sequence.generator.model.summary())
    # print(Learning_Sequence.generator.model.predict(np.ones((1, 512, 512, 1))))
    print(Learning_Sequence.discriminator1.model.summary())
    print(Learning_Sequence.discriminator2.model.summary())
