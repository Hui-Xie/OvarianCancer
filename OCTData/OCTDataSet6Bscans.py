from torch.utils import data
import numpy as np
import json
import os
import torch
import torchvision.transforms as TF

import sys
sys.path.append("../..")
from OCTData.OCTAugmentation import *


# this dataset loader load continuous 3 Bscan to predict the segmentation of the middle one.


class OCTDataSet6Bscans(data.Dataset):
    def __init__(self, imagesPath, IDPath=None, labelPath=None, transform=None, hps=None):
        super().__init__()
        self.hps = hps

        self.m_images = None
        self.m_labels = None
        self.m_IDs = None

        if imagesPath is not None:
            if self.hps.dataIn1Parcel:
                # image uses float32
                images = torch.from_numpy(np.load(imagesPath).astype(float)).to(self.hps.device, dtype=torch.float)  # slice, H, W
                # normalize images for each slice
                std,mean = torch.std_mean(images, dim=(1,2))
                self.m_images = TF.Normalize(mean, std)(images)  # standard smoothed images

                # get CLAHE images
                if hps.useCLAHEImages:
                    pathBase, ext = os.path.splitext(imagesPath)
                    claheImagePath = pathBase+"_clahe" + ext
                    claheImages = torch.from_numpy(np.load(claheImagePath).astype(float)).to(self.hps.device, dtype=torch.float)  # slice, H, W
                    std, mean = torch.std_mean(claheImages, dim=(1, 2))
                    self.m_claheImages = TF.Normalize(mean, std)(claheImages)  # CLAHE images
                else:
                    self.m_claheImages = None

                if hps.useCLAHEReplaceSmoothed:
                    self.m_images = self.m_claheImages
                    self.m_claheImages = None

            else:
                assert ((labelPath is None) and (IDPath is None))
                with open(imagesPath, 'r') as f:
                    self.m_IDs = f.readlines()
                self.m_IDs = [item[0:-1] for item in self.m_IDs]
                self.m_images = self.m_IDs.copy()
                if hps.existGTLabel:
                    self.m_labels = [item.replace("_images.npy", "_surfaces.npy") for item in self.m_images]

        if labelPath is not None:
            self.m_labels = torch.from_numpy(np.load(labelPath).astype(float)).to(self.hps.device, dtype=torch.float)  # slice, num_surface, W

        if IDPath is not None:
            with open(IDPath) as f:
                self.m_IDs = json.load(f)
        self.m_transform = transform


    def __len__(self):
        if self.hps.dataIn1Parcel:
            return self.m_images.size()[0]
        else:
            if self.hps.dataInSlice:
                return len(self.m_IDs)*self.hps.slicesPerPatient
            else:
                return len(self.m_IDs)

    def generateGradientImage(self, image, gradChannels):
        '''
        gradient should use both-side gradient approximation formula: (f_{i+1}-f_{i-1})/2,
        and at boundaries of images, use single-side gradient approximation formula: (f_{i}- f_{i-1})/2
        :param image: in size: HxW
        :param gradChannels:  integer
        :return:
        '''
        e = 1e-8
        H,W =image.shape
        device = image.device

        if self.hps.bothSideGrad:
            image_1H = image[0:-2, :]  # size: H-2,W
            image1H = image[2:, :]
            gradH = torch.cat(((image[1,:]- image[0,:]).view(1,W),
                               (image1H - image_1H)/2.0,
                               (image[-1,:]-image[-2,:]).view(1,W)), dim=0)  # size: H,W; grad90

            image_1W = image[:, 0:-2]  # size: H,W-2
            image1W = image[:, 2:]
            gradW = torch.cat(((image[:,1]- image[:,0]).view(H,1),
                               (image1W - image_1W)/2,
                               (image[:,-1]- image[:,-2]).view(H,1)), dim=1)  # size: H,W; grad0

            gradMagnitudeHW = torch.sqrt(torch.pow(gradH, 2) + torch.pow(gradW, 2))

            # normalize
            min = torch.min(gradMagnitudeHW)
            max = torch.max(gradMagnitudeHW)
            gradMagnitudeHW = (gradMagnitudeHW - min) / (max - min + e)  # in range [0,1]

            std, mean = torch.std_mean(gradH)
            gradH = (gradH - mean) / (std + e)  # in range[-1,1]

            std, mean = torch.std_mean(gradW)
            gradW = (gradW - mean) / (std + e)

            if gradChannels >= 3:
                onesImage = torch.ones_like(image)
                negOnesImage = -onesImage
                signHW = torch.where(gradH * gradW >= 0, onesImage, negOnesImage)
                gradDirectionHW = torch.atan(signHW * torch.abs(gradH) / (torch.abs(gradW) + e))
                std, mean = torch.std_mean(gradDirectionHW)
                gradDirectionHW = (gradDirectionHW - mean) / (std + e)

            if gradChannels >= 5:
                image45_0 = image[0:-2, 2:]  # size: H-2,W-2
                image45_1 = image[2:, 0:-2]  # size: H-2,W-2
                grad45 = torch.cat(((image[1:-1,0]-image[0:-2,1]).view(H-2,1),
                                    (image45_1 - image45_0)/2.0,
                                    (image[1:-1,-2]-image[0:-2,-1]).view(H-2,1)), dim=1) # size: H-2, W

                row0 = torch.cat((torch.tensor([0.0],device=device).view(1,1), (image[1,0:-1]-image[0,1:]).view(1,W-1)),dim=1)  # size: 1,W
                row_1 = torch.cat(((image[-1,0:-1]-image[-2,1:]).view(1,W-1), torch.tensor([0.0],device=device).view(1,1)),dim=1) # size:1,W
                grad45 = torch.cat((row0,
                                    grad45,
                                    row_1), dim=0)
                std, mean = torch.std_mean(grad45)
                grad45 = (grad45 - mean) / (std + e)

                image135_0 = image[0:-2, 0:-2]  # size: H-2,W-2
                image135_1 = image[2:, 2:]  # size: H-2,W-2
                grad135 = torch.cat(((image[2:, 1] - image[1:-1, 0]).view(H - 2, 1),
                                    (image135_1 - image135_0) / 2.0,
                                    (image[1:-1, -1] - image[0:-2, -2]).view(H - 2, 1)), dim=1)  # size: H-2, W

                row0 = torch.cat(
                    ((image[1, 1:] - image[0, 0:-1]).view(1, W - 1), torch.tensor([0.0], device=device).view(1, 1)),
                    dim=1)  # size: 1,W
                row_1 = torch.cat(
                    (torch.tensor([0.0], device=device).view(1, 1), (image[-1, 1:] - image[-2, 0:-1]).view(1, W - 1)),
                    dim=1)  # size:1,W
                grad135 = torch.cat((row0,
                                    grad135,
                                    row_1), dim=0)
                std, mean = torch.std_mean(grad135)
                grad135 = (grad135 - mean) / (std + e)

            if gradChannels >= 7:
                sign135_45 = torch.where(grad135 * grad45 >= 0, onesImage, negOnesImage)
                gradDirection135_45 = torch.atan(sign135_45 * torch.abs(grad135) / (torch.abs(grad45) + e))
                std, mean = torch.std_mean(gradDirection135_45)
                gradDirection135_45 = (gradDirection135_45 - mean) / (std + e)

        else: # use single Side Grad:
            image0H = image[0:-1,:]  # size: H-1,W
            image1H = image[1:,  :]
            gradH   = image1H-image0H
            gradH = torch.cat((gradH, torch.zeros((1, W), device=device)), dim=0)  # size: H,W; grad90

            image0W = image[:,0:-1]  # size: H,W-1
            image1W = image[:,1:  ]
            gradW = image1W - image0W
            gradW = torch.cat((gradW, torch.zeros((H, 1), device=device)), dim=1)  # size: H,W; grad0

            gradMagnitudeHW = torch.sqrt(torch.pow(gradH,2)+torch.pow(gradW,2))

            # normalize
            min = torch.min(gradMagnitudeHW)
            max = torch.max(gradMagnitudeHW)
            gradMagnitudeHW = (gradMagnitudeHW- min)/(max-min+e)   # in range [0,1]

            std, mean = torch.std_mean(gradH)
            gradH = (gradH - mean) / (std + e)  # in range[-1,1]

            std, mean = torch.std_mean(gradW)
            gradW = (gradW - mean) / (std + e)

            if gradChannels>=3:
                onesImage = torch.ones_like(image)
                negOnesImage = -onesImage
                signHW = torch.where(gradH * gradW >= 0, onesImage, negOnesImage)
                gradDirectionHW = torch.atan(signHW * torch.abs(gradH) / (torch.abs(gradW) + e))
                std, mean = torch.std_mean(gradDirectionHW)
                gradDirectionHW = (gradDirectionHW - mean) / (std + e)

            if gradChannels >= 5:
                image45_0 = image[0:-1,1:]  # size: H-1,W-1
                image45_1 = image[1:,0:-1]  # size: H-1,W-1
                grad45 = image45_1 - image45_0 # size: H-1,W-1
                grad45 = torch.cat((torch.zeros((H-1,1), device=device), grad45), dim=1)
                grad45 = torch.cat((grad45, torch.zeros((1, W), device=device)), dim=0)
                std, mean = torch.std_mean(grad45)
                grad45 = (grad45 - mean) / (std + e)

                image135_0 = image[0:-1, 0:-1]  # size: H-1,W-1
                image135_1 = image[1:, 1:]  # size: H-1,W-1
                grad135 = image135_1 - image135_0  # size: H-1,W-1
                grad135 = torch.cat((grad135, torch.zeros((H - 1, 1), device=device)), dim=1)
                grad135 = torch.cat((grad135, torch.zeros((1, W), device=device)), dim=0)
                std, mean = torch.std_mean(grad135)
                grad135 = (grad135 - mean) / (std + e)

            if gradChannels >= 7:
                sign135_45 = torch.where(grad135 * grad45 >= 0, onesImage, negOnesImage)
                gradDirection135_45 = torch.atan(sign135_45 * torch.abs(grad135) / (torch.abs(grad45) + e))
                std, mean = torch.std_mean(gradDirection135_45)
                gradDirection135_45 = (gradDirection135_45 - mean) / (std + e)

        # since July 23th, 2020, gradMagnitudeHW are put into the final channel
        if 1 == gradChannels:
            return gradMagnitudeHW
        elif 2 == gradChannels:
            return gradH, gradW,
        elif 3 == gradChannels:
             return gradH, gradW, gradMagnitudeHW
        elif 4 == gradChannels:
            return gradH, gradW, gradDirectionHW, gradMagnitudeHW
        elif 5 == gradChannels:
            return gradH, gradW, grad45, grad135, gradMagnitudeHW
        elif 6 == gradChannels:
            return gradH, gradW, grad45, grad135, gradDirectionHW, gradMagnitudeHW
        elif 7 == gradChannels:
            return gradH, gradW, grad45, grad135, gradDirectionHW, gradDirection135_45, gradMagnitudeHW
        else:
            print(f"Currently do not support gradChannels >7")
            assert False
            return None



    def __getitem__(self, index):
        if self.hps.dataIn1Parcel:
            label = None
            if self.m_labels is not None:
                label = self.m_labels[index,].clone() # size: N,W
            imageID = self.m_IDs[str(index)]

            #get B and s index from "_B200_s120"
            # get imageID and nB
            ID1 = int(imageID[-3:])      # "*_s003" -> 003, the middle ID.
            nB = int(imageID[-8:-5])
            s = ID1

            # get its below and up continuous Bscan by judging its imageID
            if index >0 and index< self.__len__()-1:
                # below code must use clone, otherwise original data will be contaminated.
                data = self.m_images[index-1: index+2,].clone()    # 3 smoothed Bscans.
                ID0 = int((self.m_IDs[str(index - 1)])[-3:])  # "*_s003" -> 003
                ID2 = int((self.m_IDs[str(index + 1)])[-3:])

                if index ==200:
                    print("debug")


                if ID0+1 != ID1:
                    data[0,] = data[1,]
                if ID1+1 != ID2:
                    data[2,] = data[1,]

                if self.m_claheImages != None:
                    # below code must use clone, otherwise original data will be contaminated.
                    claheData = self.m_claheImages[index - 1: index + 2, ].clone()  # 3 clahe Bscans.
                    if ID0 + 1 != ID1:
                        claheData[0,] = claheData[1,]
                    if ID1 + 1 != ID2:
                        claheData[2,] = claheData[1,]
                else:
                    claheData = None

            elif index ==0:  # replicate boundary Bscan.
                data = torch.cat((self.m_images[index].unsqueeze(dim=0), self.m_images[index: index+2]), dim=0)
                if self.m_claheImages != None:
                    claheData = torch.cat((self.m_claheImages[index].unsqueeze(dim=0), self.m_claheImages[index: index + 2]), dim=0)
                else:
                    claheData = None
            else: # index == N-1:
                data = torch.cat((self.m_images[index-1: index+1,], self.m_images[index].unsqueeze(dim=0)), dim=0)
                if self.m_claheImages != None:
                    claheData = torch.cat((self.m_claheImages[index - 1: index + 1, ], self.m_claheImages[index].unsqueeze(dim=0)), dim=0)
                else:
                    claheData = None

            if claheData != None:
                data = torch.cat((data, claheData), dim=0)
                # image order: smoothed_{i-1}, smoothed_{i}, smoothed_{i+1}, clahe_{i-1}, clahe_{i}, clahe_{i+1}
        else:
            assert False

        B, H, W = data.shape  # B ==3 or 6

        if self.m_transform: # for volume transform
            # not support rotation.
            for i in range(B):  # each slice may has different noise or no noise.
                data[i,] = self.m_transform(data[i,])

        # normalization should put outside of transform, as validation may not use transform
        # normalization should on each slice.
        std, mean = torch.std_mean(data, dim=(1,2))
        data = TF.Normalize(mean, std)(data)  # size: BxHxW

        N, W1 = label.shape
        assert W==W1
        assert N == self.hps.numSurfaces
        if ("YufanHe" not in self.hps.network) and (0 != self.hps.gradChannels):
            grads = self.generateGradientImage(data[1], self.hps.gradChannels)
        else:
            grads = None

        image = data
        if grads is not None:
            for grad in grads:
                image = torch.cat((image, grad.unsqueeze(dim=0)), dim=0)

        layerGT = []
        if self.hps.useLayerDice and label is not None:
            layerGT = getLayerLabels(label,H)

        riftWidthGT = []
        if self.hps.useRift:
            riftWidthGT = label[1:, :] - label[0:-1, :]
            if self.hps.smoothRift:
                riftWidthGT = smoothCMA(riftWidthGT, self.hps.smoothHalfWidth, self.hps.smoothPadddingMode)

        epsilon = 1.0e-8
        if self.hps.useSpaceChannels:
            bscanSpace = torch.ones((1,H,W), dtype=torch.float, device=data.device)*(s*1.0/(nB+epsilon))
            ascanSpace =  torch.arange(epsilon,1,1.0/W).view(1,1,W).expand(1,H,W).to(device=data.device)
            image = torch.cat((image,bscanSpace, ascanSpace), dim=0)
            # Right/Left Eyes: OD/OS, and OS flip to OD, so A=0 indicates temporal and A=1 indicates nasal.

        assert image.shape[0] == self.hps.inputChannels

        result = {"images": image,  #  Maybe: 3M,3C, 3M3C, 3M2S, 3C2S, 3M3C2S
                  "GTs": [] if label is None else label,
                  "gaussianGTs": [] if 0 == self.hps.sigma or label is None  else gaussianizeLabels(label, self.hps.sigma, H),
                  "IDs": imageID,
                  "layers": layerGT,
                  "riftWidth": riftWidthGT}
        return result









