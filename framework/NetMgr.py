import os
import torch
import numpy as np


class NetMgr:
    def __init__(self, net, netPath, device):
        self.m_net = net
        self.m_netPath = netPath
        if not os.path.exists(self.m_netPath):
            os.makedirs(self.m_netPath)  # for recursive make dirs.
        self.m_device = device

        if 'Best' == os.path.basename(netPath):
            self.m_netBestPath = netPath
        else:
            self.m_netBestPath = os.path.join(self.m_netPath, 'Best')
            if not os.path.exists(self.m_netBestPath):
                os.mkdir(self.m_netBestPath)

    def saveNet(self, netPath=None):
        netPath = self.m_netPath if netPath is None else netPath
        torch.save(self.m_net.state_dict(), os.path.join(netPath, "Net.pt"))
        torch.save(self.m_net.m_optimizer.state_dict(), os.path.join(netPath, "Optimizer.pt"))
        torch.save(self.m_net.m_runParametersDict, os.path.join(netPath, "ConfigParameters.pt"))

    # save real time network parameters
    def saveRealTimeNet(self, netPath=None):
        if netPath is None:
            realTimeNetPath = os.path.join(self.m_netPath, "realtime")
        else:
            realTimeNetPath = os.path.join(netPath, "realtime")
        if not os.path.exists(realTimeNetPath):
            os.makedirs(realTimeNetPath)  # for recursive make dirs.
        self.saveNet(realTimeNetPath)

    def loadNet(self, mode):
        # Save on GPU, Load on GPU
        self.m_net.load_state_dict(torch.load(os.path.join(self.m_netPath, "Net.pt"), map_location=self.m_device))
        if os.path.exists(os.path.join(self.m_netPath, "ConfigParameters.pt")):
            self.m_net.m_runParametersDict = torch.load(os.path.join(self.m_netPath, "ConfigParameters.pt"))
        if mode == "train":
            # Moves all model parameters and buffers to the GPU.So it should be called before constructing optimizer if the module will live on GPU while being optimized.
            self.m_net.m_optimizer.load_state_dict(torch.load(os.path.join(self.m_netPath, "Optimizer.pt")))
            self.m_net.train()
        elif mode == "test":   # eval
            self.m_net.eval()
        else:
            print("Error: loadNet mode is incorrect.")

    # Deprecated as configParameterDict can save any config parameter.
    def saveBestTestPerf(self, testPerf, netPath=None):
        netPath = self.m_netPath if netPath is None else netPath
        testPerfArray = np.asarray(testPerf)
        np.save(os.path.join(netPath, "bestTestPerf.npy"), testPerfArray)

    # Deprecated as configParameterDict can save any config parameter.
    def loadBestTestPerf(self, K=1):
        filename = os.path.join(self.m_netPath, "bestTestPerf.npy")
        if os.path.isfile(filename):
            bestTestPerf = np.load(filename)
            if K >1:
                bestTestPerf.tolist()
            else:
                bestTestPerf = bestTestPerf.item(0)
        else:
            if K>1:
                bestTestPerf = [0]*K   # for 3 classifications
            else:
                bestTestPerf = 0
        return bestTestPerf

    # Deprecated as configParameterDict can save any config parameter.
    def saveBest(self, testPerf, netPath=None):
        netPath = self.m_netBestPath if netPath is None else netPath
        self.save(testPerf, netPath)

    # Deprecated as configParameterDict can save any config parameter.
    def save(self, testPerf, netPath=None):
        netPath = self.m_netPath if netPath is None else netPath
        self.saveNet(netPath)
        self.saveBestTestPerf(testPerf, netPath)
