from torch.utils import data
import numpy as np
import json
import os
import torch
import torchvision.transforms as TF

from network.OCTAugmentation import *


class OCTDataSet(data.Dataset):
    def __init__(self, imagesPath, IDPath=None, labelPath=None, transform=None, hps=None):
        self.hps = hps

        self.m_images = None
        self.m_labels = None
        self.m_IDs = None

        if imagesPath is not None:
            if self.hps.dataIn1Parcel:
                # image uses float32
                images = torch.from_numpy(np.load(imagesPath).astype(np.float32)).to(self.hps.device, dtype=torch.float)  # slice, H, W
                # normalize images for each slice
                std,mean = torch.std_mean(images, dim=(1,2))
                self.m_images = TF.Normalize(mean, std)(images)
            else:
                assert ((labelPath is None) and (IDPath is None))
                with open(imagesPath, 'r') as f:
                    self.m_IDs = f.readlines()
                self.m_IDs = [item[0:-1] for item in self.m_IDs]
                self.m_images = self.m_IDs.copy()
                self.m_labels = [item.replace("_images.npy", "_surfaces.npy") for item in self.m_images]

        if labelPath is not None:
            self.m_labels = torch.from_numpy(np.load(labelPath).astype(np.float32)).to(self.hps.device, dtype=torch.float)  # slice, num_surface, W

        if IDPath is not None:
            with open(IDPath) as f:
                self.m_IDs = json.load(f)
        self.m_transform = transform


    def __len__(self):
        if self.hps.dataIn1Parcel:
            return self.m_images.size()[0]
        else:
            return len(self.m_IDs)*self.hps.slicesPerPatient

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
            data = self.m_images[index,]

            label = None
            if self.m_labels is not None:
                label = self.m_labels[index,] # size: N,W
            imageID = self.m_IDs[str(index)]
        else:
            volumeIndex = index // self.hps.slicesPerPatient
            offset = index % self.hps.slicesPerPatient

            if self.hps.dataInSlice:  # one slice saved as one file
                imagesRoot, imageExt = os.path.splitext(self.m_IDs[volumeIndex])
                imageID = imagesRoot + f"_s{offset:02d}" + imageExt
                lableID = imageID.replace("_images_", "_surfaces_")
                data = torch.from_numpy(np.load(imageID).astype(np.float32)).to(self.hps.device, dtype=torch.float)  # H, W
                label = torch.from_numpy(np.load(lableID).astype(np.float32)).to(self.hps.device, dtype=torch.float) # N, W

            else:

                # image uses float32
                images = torch.from_numpy(np.load(self.m_images[volumeIndex]).astype(np.float32)).to(self.hps.device, dtype=torch.float)  # slice, H, W
                # normalize images for each slice, which has done in converting from mat to numpy
                # std, mean = torch.std_mean(images, dim=(1, 2))
                # images = TF.Normalize(mean, std)(images)
                data = images[offset,]

                labels = torch.from_numpy(np.load(self.m_labels[volumeIndex]).astype(np.float32)).to(self.hps.device, dtype=torch.float)  # slice, num_surface, W
                label = labels[offset,]

                imageID = self.m_IDs[volumeIndex]+f".OCT{offset:2d}"

        if self.m_transform:
            data, label = self.m_transform(data, label)

        if self.hps.TTA and 0 != self.hps.TTA_Degree:
            data, label = polarImageLabelRotate_Tensor(data, label, rotation=self.hps.TTA_Degree)

        if 0 != self.hps.lacingWidth:
            data, label = lacePolarImageLabel(data,label,self.hps.lacingWidth)

        if 1 != self.hps.scaleNumerator or 1 != self.hps.scaleDenominator:  # this will change the Height of polar image
            data = scalePolarImage(data, self.hps.scaleNumerator, self.hps.scaleDenominator)
            label = scalePolarLabel(label, self.hps.scaleNumerator, self.hps.scaleDenominator)

        H, W = data.shape
        N, W1 = label.shape
        assert W==W1
        image = data.unsqueeze(dim=0)
        if 0 != self.hps.gradChannels:
            grads = self.generateGradientImage(data, self.hps.gradChannels)
            for grad in grads:
                image = torch.cat((image, grad.unsqueeze(dim=0)),dim=0)

        layerGT = []
        if self.hps.useLayerDice and label is not None:
            layerGT = getLayerLabels(label,H)

        riftWidthGT = []
        # N rifts for N surfaces
        #riftWidthGT = torch.cat((label[0,:].unsqueeze(dim=0),label[1:,:]-label[0:-1,:]),dim=0)
        # (N-1) rifts for N surfaces.
        riftWidthGT = label[1:, :] - label[0:-1, :]
        if self.hps.smoothRift:
            riftWidthGT = smoothCMA(riftWidthGT, self.hps.smoothHalfWidth, self.hps.smoothPadddingMode)

        result = {"images": image,
                  "GTs": [] if label is None else label,
                  "gaussianGTs": [] if 0 == self.hps.sigma or label is None  else gaussianizeLabels(label, self.hps.sigma, H),
                  "IDs": imageID,
                  "layers": layerGT,
                  "riftWidth": riftWidthGT}
        return result









