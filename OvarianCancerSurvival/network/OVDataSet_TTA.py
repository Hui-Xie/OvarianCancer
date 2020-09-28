# DataSet support TTA

from torch.utils import data
import numpy as np
import random
import torch
import SimpleITK as sitk
from OVTools import readGTDict8Cols, readGTDict6Cols


class OVDataSet_TTA(data.Dataset):
    def __init__(self, mode, hps=None, transform=None):
        '''
        Ovarian Cancer data set Manager
        :param mode: training, validation, test
        :param hps:
        :param transform:
        '''
        super().__init__()
        self.m_mode = mode
        self.hps = hps

        if mode == "training":
            IDPath = hps.trainingDataPath
            gtPath = hps.trainingGTPath
        elif mode == "validation":
            IDPath = hps.validationDataPath
            gtPath = hps.validationGTPath
        elif mode == "test":
            IDPath = hps.testDataPath
            gtPath = hps.testGTPath
        else:
            assert False
            print(f"OVDataSet mode error")

        self.m_imagesPath = hps.dataDir

        with open(IDPath, 'r') as idFile:
            MRNList = idFile.readlines()
        MRNList = [item[0:-1] for item in MRNList]  # erase '\n'
        MRNList = ['0'+item if (len(item) == 7) else item  for item in MRNList]
        self.m_IDs = MRNList

        if 8 == hps.colsGT:
            self.m_labels = readGTDict8Cols(gtPath)
        else:
            self.m_labels = readGTDict6Cols(gtPath)

        assert (self.m_transform == None)
        self.m_edgeCropRate = math.sqrt(hps.randomCropArea)



    def getGTDict(self):
        return self.m_labels

    def addVolumeGradient(self, volume):
        '''
        gradient should use both-side gradient approximation formula: (f_{i+1}-f_{i-1})/2,
        and at boundaries of images, use single-side gradient approximation formula: (f_{i}- f_{i-1})/2
        :param volume: in size: SxHxW
        :param gradChannels:  integer
        :return:
                Bx3xHxW, added gradient volume without normalization
        '''
        S,H,W =volume.shape

        image_1H = volume[:, 0:-2, :]  # size: S, H-2,W
        image1H = volume[:, 2:, :]
        gradH = torch.cat(((volume[:, 1, :] - volume[:, 0, :]).view(S, 1, W),
                           (image1H - image_1H)/2.0,
                           (volume[:, -1, :] - volume[:, -2, :]).view(S, 1, W)), dim=1)  # size: S, H,W; grad90

        image_1W = volume[:,:, 0:-2]  # size: S, H,W-2
        image1W = volume[:, :, 2:]
        gradW = torch.cat(((volume[:,:, 1] - volume[:,:, 0]).view(S, H, 1),
                           (image1W - image_1W)/2,
                           (volume[:,:, -1] - volume[:,:, -2]).view(S, H, 1)), dim=2)  # size: S, H,W; grad0

        # concatenate
        gradVolume = torch.cat((volume.unsqueeze(dim=1), gradH.unsqueeze(dim=1), gradW.unsqueeze(dim=1)), dim=1) # B,3,H,W

        return gradVolume


    def generator(self):
        epsilon = 1e-8
        device = self.hps.device

        H = self.hps.imageH
        W = self.hps.imageW
        newH = int(H * self.m_edgeCropRate)
        newW = int(W * self.m_edgeCropRate)
        gapH = H - newH
        gapW = W - newW

        for MRN in self.m_IDs:
            labels= []
            if self.hps.existGTLabel:
                labels = self.m_labels[MRN]

            volumePath = self.m_imagesPath+"/" +MRN+"_CT.npy"
            npVolume = np.load(volumePath)

            data = torch.from_numpy(npVolume).to(device)
            S,_,_ = data.shape

            # Sample a fixed N slices
            N = self.hps.sampleSlicesPerPatient
            sectionLen = S//N
            sampleSlices = []
            for i in range(N):
                start = i*sectionLen
                end = (i+1)*sectionLen
                if end >= S:
                    end = S-1
                if self.hps.randomSliceSample:
                    sampleSlices.append(random.choice(list(range(start,end))))
                else:
                    sampleSlices.append((start+end)//2)  # fixed slice sample
            data = data[sampleSlices,:,:]

            # 10 crops:
            tenCrops = torch.empty((10, self.hps.inputChannels, newH, newW), device=device, dtype=torch.float32)
            tenCrops[0,] = data[:,gapH//2:gapH//2+newH, gapW//2: gapW//2+newW]  # center crop
            tenCrops[1,] = data[:, 0:newH, 0: newW]
            tenCrops[2,] = data[:, 0:newH, gapW: gapW+newW]
            tenCrops[3,] = data[:, gapH:gapH+newH, 0: newW]
            tenCrops[4,] = data[:, gapH:gapH+newH, gapW: gapW + newW]
            wFlipData = torch.flip(data, [2])
            tenCrops[5,] = wFlipData[:, gapH // 2:gapH // 2 + newH, gapW // 2: gapW // 2 + newW]  # center crop
            tenCrops[6,] = wFlipData[:, 0:newH, 0: newW]
            tenCrops[7,] = wFlipData[:, 0:newH, gapW: gapW + newW]
            tenCrops[8,] = wFlipData[:, gapH:gapH + newH, 0: newW]
            tenCrops[9,] = wFlipData[:, gapH:gapH + newH, gapW: gapW + newW]

            if 0 != self.hps.gradChannels:
                data = self.addVolumeGradient(data)

            # normalization before output to dataloader
            # AlexNex, GoogleNet V1, VGG, ResNet only do mean subtraction without dividing std.
            #mean = torch.mean(data, dim=(-1, -2), keepdim=True)
            #mean = mean.expand_as(data)
            #data = data - mean

            # Normalization
            std, mean = torch.std_mean(tenCrops, dim=(-1, -2), keepdim=True)
            std = std.expand_as(tenCrops)
            mean = mean.expand_as(tenCrops)
            tenCrops = (tenCrops - mean) / (std + epsilon)  # size: Bx3xHxW

            # replicate label
            B,_,_,_ = tenCrops.shape
            for k,v in labels.items():
                labels[k] = [v,]*B

            result = {"images": tenCrops,
                      "GTs": labels,
                      "IDs": [MRN,]*B
                     }
            yield  result










