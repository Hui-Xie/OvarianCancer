from torch.utils import data
import numpy as np
import os
import torch
import torchvision.transforms as TF
import csv
import SimpleITK as sitk

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

        self.m_imagesPath = hps.dataDir+"/nrrd"

        with open(IDPath, 'r') as idFile:
            MRNList = idFile.readlines()
        MRNList = [item[0:-1] for item in MRNList]  # erase '\n'
        MRNList = ['0'+item if (len(item) == 7) else item  for item in MRNList]
        self.m_IDs = MRNList

        '''
        csv data example:
        MRN,Age,ResidualTumor,Censor,TimeSurgeryDeath(d),ChemoResponse
        3818299,68,0,1,316,1
        5723607,52,0,1,334,0
        68145843,70,0,1,406,0
        4653841,64,0,1,459,0
        96776044,49,0,0,545,1

        '''
        gtDict = {}
        with open(gtPath, newline='') as csvfile:
            csvList = list(csv.reader(csvfile, delimiter=',', quotechar='|'))
            csvList = csvList[1:]  # erase table head
            for row in csvList:
                lengthRow = len(row)
                MRN = '0' + row[0] if 7 == len(row[0]) else row[0]
                gtDict[MRN] = {}
                gtDict[MRN]['Age'] = int(row[1])
                gtDict[MRN]['ResidualTumor'] = int(row[2])
                gtDict[MRN]['Censor'] = int(row[3]) if 0 != len(row[3]) else None
                gtDict[MRN]['SurvivalMonths'] = int(row[4]) / 30.4368 if 0 != len(row[4]) else None
                gtDict[MRN]['ChemoResponse'] = int(row[5]) if 0 != len(row[5]) else None
        self.m_labels = gtDict

        self.m_transform = transform


    def __len__(self):
        return len(self.m_IDs)

    def addVolumeGradient(self, volume):
        '''
        gradient should use both-side gradient approximation formula: (f_{i+1}-f_{i-1})/2,
        and at boundaries of images, use single-side gradient approximation formula: (f_{i}- f_{i-1})/2
        :param volume: in size: SxHxW
        :param gradChannels:  integer
        :return:
                Bx3xHxW, added gradient volume
        '''
        e = 1e-8
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

        # normalize
        std, mean = torch.std_mean(gradH, dim=(1,2), keepdim=True).expand_as(gradH)
        gradH = (gradH - mean) / (std + e)  # in range[-1,1]

        std, mean = torch.std_mean(gradW, dim=(1,2), keepdim=True).expand_as(gradW)
        gradW = (gradW - mean) / (std + e)

        # concatenate
        gradVolume = torch.cat((volume.unsqueeze(dim=1), gradH.unsqueeze(dim=1), gradW.unsqueeze(dim=1)), dim=1) # B,3,H,W

        return gradVolume


    def __getitem__(self, index):
        MRN = self.m_IDs[index]

        labels= []
        if self.hps.existGTLabel:
            labels = self.m_labels[MRN]

        volumePath = self.hps.dataDir+"/" +MRN+"_CT.nrrd"
        itkImage = sitk.ReadImage(volumePath)
        npVolume = sitk.GetArrayFromImage(itkImage)
        _, H, W = npVolume.shape

        if self.m_transform:
            data = self.m_transform(npVolume)

        if 0 != self.hps.gradChannels:
            data = self.addVolumeGradient(data)

        result = {"images": data,
                  "GTs": labels,
                  "IDs": MRN
                 }
        return result









