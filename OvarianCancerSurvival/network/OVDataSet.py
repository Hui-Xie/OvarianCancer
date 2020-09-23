from torch.utils import data
import numpy as np
import random
import torch
import SimpleITK as sitk
from OVTools import readGTDict8Cols, readGTDict6Cols


class OVDataSet(data.Dataset):
    def __init__(self, mode, hps=None, transform=None, ):
        '''
        Ovarian Cancer data set Manager
        :param mode: training, validation, test
        :param hps:
        :param transform:
        '''
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

        self.m_transform = transform

    def __len__(self):
        return len(self.m_IDs)

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


    def __getitem__(self, index):
        epsilon = 1e-8
        MRN = self.m_IDs[index]

        labels= []
        if self.hps.existGTLabel:
            labels = self.m_labels[MRN]

        volumePath = self.m_imagesPath+"/" +MRN+"_CT.nrrd"
        itkImage = sitk.ReadImage(volumePath)
        npVolume = sitk.GetArrayFromImage(itkImage).astype(dtype=np.float32)
        data = torch.from_numpy(npVolume).to(self.hps.device)
        S,H,W = data.shape

        # scale down 1/2 in H and W respectively
        data = data[:, 0:-1:2, 0:-1:2]

        # random sample a fixed N slices
        N = self.hps.sampleSlicesPerPatient
        sectionLen = S//N
        sampleSlices = []
        for i in range(N):
            start = i*sectionLen
            end = (i+1)*sectionLen
            if end > S:
                end = S
            sampleSlices.append(random.choice(list(range(start,end))))
        data = data[sampleSlices,:,:]

        # sample
        if self.m_transform:
            data = self.m_transform(data)  # size: BxHxW

        if 0 != self.hps.gradChannels:
            data = self.addVolumeGradient(data)

        # normalization before output to dataloader
        std, mean = torch.std_mean(data, dim=(-1, -2), keepdim=True)
        std = std.expand_as(data)
        mean = mean.expand_as(data)
        data = (data - mean) / (std + epsilon)  # size: Bx3xHxW

        result = {"images": data,
                  "GTs": labels,
                  "IDs": MRN
                 }
        return result









