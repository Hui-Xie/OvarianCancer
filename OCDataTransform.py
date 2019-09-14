# data transform for Ovarian Cancer

import torchvision.transforms.functional as TF
import random
import PIL
import torch
import math

# notes: from nrrd to numpy, the patient data have performed window level shreshold, [0,1] normalization with non-zero mean, and 0-padding to [231,251,251]

class OCDataTransform(object):
    def __init__(self, prob =0):
        self.m_prob = prob

    def __call__(self, data):
        d,h,w = data.shape

        # specific parameters of affine transform for each slice
        while True:
            if random.uniform(0, 1) < self.m_prob:
                affine = True
                angle = random.randrange(-180, 180, 10)
                translate = random.randrange(-38, 39, 3), random.randrange(-38, 39, 3)  # 15% of maxsize of Y, X
                scale = random.uniform(0.6, 1.25)
                shear = random.randrange(-30, 31, 10)  #90 degree is too big, which almost compresses image into a line.
            else:
                affine = False
                angle = 0
                translate = 0,0
                scale = 1
                shear = 0

            angle1 = math.radians(angle)
            shear1 = math.radians(shear)
            dInAffine = math.cos(angle1 + shear1) * math.cos(angle1) + math.sin(angle1 + shear1) * math.sin(angle1)
            if  0 != dInAffine:
                break

        # create output tensor
        outputTensor = torch.zeros((d,h,w), dtype=torch.float32)

        for z in range(d):
            slice = data[z,]  # normalized array
            # affine transform
            if affine:
                # to PIL image
                slice = TF.to_pil_image(slice)
                slice = TF.affine(slice, angle, translate, scale, shear, resample=PIL.Image.BILINEAR, fillcolor=0)
            # to tensor
            outputTensor[z,] = TF.to_tensor(slice)
        return outputTensor

    def __repr__(self):
        return self.__class__.__name__

class OCDataLabelTransform(object):
    def __init__(self, prob =0):
        self.m_prob = prob

    def __call__(self, data, label):
        assert data.shape == label.shape
        d,h,w = data.shape

        # specific parameters of affine transform for each slice.
        # todo: think to use gaussion to replace randrange int the future
        while True:
            if random.uniform(0, 1) < self.m_prob:
                affine = True
                angle = random.randrange(-180, 180, 10)
                translate = random.randrange(-25, 26, 3), random.randrange(-25, 26, 3)  # 15% of maxsize of Y, X
                scale = random.uniform(0.6, 1.25)
                shear = random.randrange(-20, 21, 5)  #90 degree is too big, which almost compresses image into a line.
                zShift = random.randrange(-7, 8, 3)  # 15% of maxSize of Z
            else:
                affine = False
                angle = 0
                translate = 0,0
                scale = 1
                shear = 0
                zShift = 0

            angle1 = math.radians(angle)
            shear1 = math.radians(shear)
            dInAffine = math.cos(angle1 + shear1) * math.cos(angle1) + math.sin(angle1 + shear1) * math.sin(angle1)
            if  0 != dInAffine:
                break

        # create output data and label
        outputData = torch.zeros((d,h,w), dtype=torch.float32)
        outputLabel = torch.zeros((d, h, w), dtype=torch.float32)
        
        for z in range(d):
            dataSlice = data[z,]  # normalized array
            labelSlice = label[z,]
            # affine transform
            if affine:
                # to PIL image
                dataSlice = TF.to_pil_image(dataSlice)
                dataSlice = TF.affine(dataSlice, angle, translate, scale, shear, resample=PIL.Image.BILINEAR, fillcolor=0)

                labelSlice = TF.to_pil_image(labelSlice)
                labelSlice = TF.affine(labelSlice, angle, translate, scale, shear, resample=PIL.Image.NEAREST, fillcolor=0)
                
            # to tensor
            outputData[z,] = TF.to_tensor(dataSlice)
            outputLabel[z,] = TF.to_tensor(labelSlice)

        # roll along z direction
        if affine and zShift != 0:
            outputData = torch.roll(outputData, zShift, dims=0)
            outputLabel = torch.roll(outputLabel, zShift, dims=0)

        return outputData, outputLabel

    def __repr__(self):
        return self.__class__.__name__
