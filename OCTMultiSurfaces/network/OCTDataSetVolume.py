from torch.utils import data
import numpy as np
import json
import os
import torch
import torchvision.transforms as TF
import glob

from network.OCTAugmentation import *

# input volume one by one.

class OCTDataSetVolume(data.Dataset):
    def __init__(self, imagesPath, IDPath=None, labelPath=None, transform=None, hps=None):
        super().__init__()
        self.hps = hps

        self.m_imagesList = None
        self.m_labels = None
        self.m_IDs = None

        if imagesPath is not None:
            self.m_imagesList = glob.glob(imagesPath + f"/*_Volume.npy")
            self.m_imagesList.sort()

        if labelPath is not None:
            self.m_labels = torch.from_numpy(np.load(labelPath).astype(np.float32)).to(self.hps.device, dtype=torch.float)  # slice, num_surface, W

        if IDPath is not None:
            with open(IDPath) as f:
                self.m_IDs = json.load(f)
        self.m_transform = transform


    def __len__(self):
        return len(self.m_imagesList)

    # deprecated
    def generateGradientImage(self, image, gradChannels):
        '''
        This is version match with SurfacesUnet0512 version.
        :param image:
        :param gradChannels:
        :return:
        '''
        H,W =image.shape
        device = image.device
        image0H = image[0:-1,:]  # size: H-1,W
        image1H = image[1:,  :]
        gradH   = image1H-image0H
        gradH = torch.cat((gradH, torch.zeros((1, W), device=device)), dim=0)  # size: H,W

        image0W = image[:,0:-1]  # size: H,W-1
        image1W = image[:,1:  ]
        gradW = image1W - image0W
        gradW = torch.cat((gradW, torch.zeros((H, 1), device=device)), dim=1)  # size: H,W

        gradMagnitude = torch.sqrt(torch.pow(gradH,2)+torch.pow(gradW,2))

        if gradChannels>=3:
            onesHW = torch.ones_like(image)
            negOnesHW = -onesHW
            signHW = torch.where(gradH * gradW >= 0, onesHW, negOnesHW)
            e = 1e-8
            gradDirection = torch.atan(signHW * torch.abs(gradH) / (torch.abs(gradW) + e))

        if 1 == gradChannels:
            return gradMagnitude
        elif 2 == gradChannels:
            return gradH, gradW,
        elif 3 == gradChannels:
             return gradH, gradW, gradDirection
        elif 4 == gradChannels:
            return gradH, gradW, gradMagnitude, gradDirection
        else:
            print(f"Currently do not support gradChannels >4")
            assert False
            return None

    def generateGradientImage_new(self, image, gradChannels):
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
        volumePath = self.m_imagesList[index]
        data = torch.from_numpy(np.load(volumePath).astype(np.float32)).to(self.hps.device, dtype=torch.float)
        B,H,W = data.shape

        # normalize images for each slice
        # its trained network did this normalization at load parcel data.
        std, mean = torch.std_mean(data, dim=(1, 2))
        data = TF.Normalize(mean, std)(data)

        basename = os.path.basename(volumePath)
        volumeName, ext = os.path.splitext(basename)
        imageID = volumeName

        label = None

        newData = torch.empty((B, self.hps.inputChannels, H,W), dtype=torch.float, device=self.hps.device)

        for b in range(B):
            image = data[b,]
            if 0 != self.hps.gradChannels:
                # for 20201125 best Tongren network
                grads = self.generateGradientImage_new(image, self.hps.gradChannels)
                image = image.unsqueeze(dim=0)
                for grad in grads:
                    image = torch.cat((image, grad.unsqueeze(dim=0)),dim=0)
            newData[b,] = image

        data = newData

        layerGT = []
        riftWidthGT = []

        result = {"images": data,
                  "GTs": [] if label is None else label,
                  "gaussianGTs": [],
                  "IDs": imageID,
                  "layers": layerGT,
                  "riftWidth": riftWidthGT}
        return result









