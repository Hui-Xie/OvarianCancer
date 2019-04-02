import os
import torch


class NetMgr:
    def __init__(self, net):
        self.m_net = net

    def saveNet(self, netPath):
        print("Program starts to save network ......")
        torch.save(self.m_net.state_dict(), os.path.join(netPath,"Net.pt"))
        torch.save(self.m_net.m_optimizer.state_dict(), os.path.join(netPath,"Optimizer.pt"))
        print("Program finished saving network.")

    def loadNet(self,netPath, forTrain):
        print(f'Program laod net from {netPath}.')
        device = torch.device('cpu')
        self.m_net.load_state_dict( torch.load(os.path.join(netPath,"Net.pt")), map_location=device)
        self.m_net.m_optimizer.load_state_dict( torch.load(os.path.join(netPath,"Optimizer.pt")), map_location=device)
        if forTrain:
            self.m_net.train()
        else: # eval
            self.m_net.eval()
