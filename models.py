import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size=256, n_actions=6, n_layers=2, fc_size=512):
        """
        Initialize MLP Network for bisim

        Args:
            in_channels (int): number of input channels
            n_actions (int): number of outputs
        """
        super(MLP, self).__init__()
        self.n_layers = n_layers

        self.fcs = nn.ModuleList([nn.Linear(input_size, fc_size)])
        for idx in range(n_layers - 1):
            self.fcs.append(nn.Linear(fc_size, fc_size))

        self.final = nn.Linear(fc_size, n_actions)
        
    def forward(self, x):
        # x = x.float() / 255
        # print("DQN: ", x.size())
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        for idx in range(self.n_layers):
            x = F.relu(self.fcs[idx](x))

        return self.final(x)

class MLP_duel(nn.Module):
    def __init__(self, input_size=256, n_actions=6, n_layers=2, fc_size=512):
        """
        Initialize MLP Network for bisim

        Args:
            in_channels (int): number of input channels
            n_actions (int): number of outputs
        """
        super(MLP_duel, self).__init__()
        self.n_layers = n_layers

        self.fcs = nn.ModuleList([nn.Linear(input_size, fc_size)])
        for idx in range(n_layers - 1):
            self.fcs.append(nn.Linear(fc_size, fc_size))

        self.fc_a = nn.Linear(fc_size, fc_size)
        self.final_a = nn.Linear(fc_size, n_actions)
        self.fc_v = nn.Linear(fc_size, fc_size)
        self.final_v = nn.Linear(fc_size, 1)
        
    def forward(self, x):
        # x = x.float() / 255
        # print("DQN: ", x.size())
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        for idx in range(self.n_layers):
            x = F.relu(self.fcs[idx](x))
        a = F.relu(self.fc_a(x))
        a = self.final_a(a)
        v = F.relu(self.fc_v(x))
        v = self.final_v(v)
        return v+a-torch.mean(a,dim=1,keepdim=True)


class DQN(nn.Module):
    def __init__(self, in_channels=4, n_actions=6):
        """
        Initialize Deep Q Network

        Args:
            in_channels (int): number of input channels
            n_actions (int): number of outputs
        """
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        # self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.final = nn.Linear(512, n_actions)
        
    def forward(self, x):
        # x = x.float() / 255
        print("DQN: ", x.size())
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        return self.final(x)


class DQN_duel(nn.Module):
    def __init__(self, in_channels=4, n_actions=6):
        """
        Initialize Deep Q Network

        Args:
            in_channels (int): number of input channels
            n_actions (int): number of outputs
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        # self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # self.bn3 = nn.BatchNorm2d(64)
        self.fc1_a = nn.Linear(7 * 7 * 64, 512)
        self.final_a = nn.Linear(512, n_actions)
        self.fc1_v=nn.Linear(7*7*64,512)
        self.final_v = nn.Linear(512,1)

    def forward(self, x):
        x = x.float() / 255
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        a = F.relu(self.fc1_a(x.view(x.size(0), -1)))
        a = self.final_a(a)
        v = F.relu(self.fc1_v(x.view(x.size(0), -1)))
        v = self.final_v(v)
        return v+a-torch.mean(a,dim=1,keepdim=True)