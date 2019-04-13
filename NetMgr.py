import os
import torch
import numpy as np


class NetMgr:
    def __init__(self, net, netPath):
        self.m_net = net
        self.m_netPath = netPath
        self.m_netBestPath = os.path.join(netPath, 'Best')
        if not os.path.exists(self.m_netBestPath):
            os.mkdir(self.m_netBestPath)

    def saveNet(self, netPath=None):
        netPath = self.m_netPath if netPath is None else netPath
        torch.save(self.m_net.state_dict(), os.path.join(netPath, "Net.pt"))
        torch.save(self.m_net.m_optimizer.state_dict(), os.path.join(netPath, "Optimizer.pt"))

    def loadNet(self, isTrain):
        print(f'Program loads net from {self.m_netPath}.')
        # Save on GPU, Load on CPU
        device = torch.device('cpu')
        self.m_net.load_state_dict(torch.load(os.path.join(self.m_netPath, "Net.pt"), map_location=device))
        if isTrain:
            # Moves all model parameters and buffers to the GPU.So it should be called before constructing optimizer if the module will live on GPU while being optimized.
            self.m_net.cuda()
            self.m_net.m_optimizer.load_state_dict(torch.load(os.path.join(self.m_netPath, "Optimizer.pt")))
            self.m_net.train()
        else:   # eval
            self.m_net.eval()

    def saveBestTestDice(self, testDiceList, netPath=None):
        netPath = self.m_netPath if netPath is None else netPath
        testDiceArray = np.asarray(testDiceList)
        np.save(os.path.join(netPath, "bestTestDice.npy"), testDiceArray)

    def loadBestTestDice(self):
        filename = os.path.join(self.m_netPath, "bestTestDice.npy")
        if os.path.isfile(filename):
            bestTestDiceList = np.load(filename)
            bestTestDiceList.tolist()
        else:
            bestTestDiceList = [0]*4
        return bestTestDiceList

    def saveBest(self, testDiceList, netPath=None):
        netPath = self.m_netBestPath if netPath is None else netPath
        self.save(testDiceList, netPath)

    def save(self, testDiceList, netPath=None):
        netPath = self.m_netPath if netPath is None else netPath
        self.saveNet(netPath)
        self.saveBestTestDice(testDiceList, netPath)
