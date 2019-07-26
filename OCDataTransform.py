# data transform for Ovarian Cancer

import torchvision.transforms.functional as TF
import random
import PIL
import torch

class OCDataTransform(object):
    def __init__(self, depth, height, width, prob =0):
        self.m_depth = depth
        self.m_height = height
        self.m_width = width
        self.m_prob = prob


    def __call__(self, data):
        d,h,w = data.shape
        zoffset = (self.m_depth -d)//2

        # specific parameters of affine transform
        affine = False
        if random.uniform(0, 1) < self.m_prob:
            affine = True
            angle = random.randrange(-180, 180)
            translate = random.randrange(-25, 25), random.randrange(-25, 25)  # 10% of maxsize
            scale = random.uniform(1, self.m_height/h)
            shear = random.randrange(-90, 90)
        else:
            angle = 0
            translate = 0,0
            scale = 1
            shear = 0
        # create output tensor
        outputTensor = torch.zeros((self.m_depth, self.m_height, self.m_width))

        for z in range(d):
            slice = data[z,]
            # normalize
            slice = slice - slice.mean()
            slice = slice/slice.max()
            # to PIL image
            slice = TF.to_pil_image(slice)
            # padding
            padding = (self.m_width- w)//2, (self.m_height - h)//2, self.m_width- (self.m_width- w)//2 -w, self.m_height - (self.m_height - h)//2 -h
            slice = TF.pad(slice, padding, fill=0, padding_mode='constant')
            # affine transform
            if affine:
                slice = TF.affine(slice, angle, translate, scale, shear, resample=PIL.Image.BILINEAR, fillcolor=0)
            # to tensor
            slice = TF.to_tensor(slice)
            outputTensor[z+zoffset,] = slice
        return outputTensor

    def __repr__(self):
        return self.__class__.__name__

