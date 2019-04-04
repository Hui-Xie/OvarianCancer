import os
import torch


class NetMgr:
    def __init__(self, net):
        self.m_net = net

    def saveNet(self, netPath):
        torch.save(self.m_net.state_dict(), os.path.join(netPath,"Net.pt"))
        torch.save(self.m_net.m_optimizer.state_dict(), os.path.join(netPath,"Optimizer.pt"))

    def loadNet(self,netPath, isTrain):
        print(f'Program loads net from {netPath}.')
        # Save on GPU, Load on CPU
        device = torch.device('cpu')
        self.m_net.load_state_dict( torch.load(os.path.join(netPath,"Net.pt"), map_location=device))

        # Moves all model parameters and buffers to the GPU.So it should be called before constructing optimizer if the module will live on GPU while being optimized.
        self.m_net.cuda()

        self.m_net.m_optimizer.load_state_dict( torch.load(os.path.join(netPath,"Optimizer.pt")))
        if isTrain:
            self.m_net.train()
        else: # eval
            self.m_net.eval()
