# data transform for Ovarian Cancer

import torchvision.transforms.functional as TF
import random
import PIL
import torch
import math

# notes: from nrrd to numpy, the patient data have performed window level shreshold, [0,1] normalization, and 0-padding to [231,251,251]

class OCDataTransform(object):
    def __init__(self, prob =0):
        self.m_prob = prob

    def __call__(self, data):
        d,h,w = data.shape

        # specific parameters of affine transform
        while True:
            if random.uniform(0, 1) < self.m_prob:
                affine = True
                angle = random.randrange(-180, 180, 10)
                translate = random.randrange(-38, 38, 3), random.randrange(-38, 38, 3)  # 15% of maxsize of Y, X
                scale = 1.0 # do not scale, random.uniform(1, self.m_height/h)
                shear = random.randrange(-90, 90, 10)
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
        outputTensor = torch.zeros((d,h,w))

        for z in range(d):
            slice = data[z,]  # float[0,1] ndarray
            # affine transform
            if affine:
                # to PIL image
                slice = TF.to_pil_image(slice)
                slice = TF.affine(slice, angle, translate, scale, shear, resample=PIL.Image.BILINEAR, fillcolor=0)
            # to tensor
            slice = TF.to_tensor(slice)
            outputTensor[z,] = slice
        return outputTensor

    def __repr__(self):
        return self.__class__.__name__

