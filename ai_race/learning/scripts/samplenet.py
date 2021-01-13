import torch
import torch.nn as nn
import torch.nn.functional as F

class SampleNet(nn.Module):
    def __init__(self):
        super(SampleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(16, 16, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(16, 16, 3, 1, padding=1)
        self.conv5 = nn.Conv2d(16, 16, 3, 1, padding=1)
        self.fc1 = nn.Linear(76800, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = nn.MaxPool2d(kernel_size=2)(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = nn.MaxPool2d(kernel_size=2)(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class SimpleNet(nn.Module):
    def __init__(self, init_maxpool=1, use_gap=True):
        super(SimpleNet, self).__init__()
        self.init_maxpool = init_maxpool
        self.use_gap = use_gap

        self.conv1 = nn.Conv2d(3, 16, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(16, 16, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(16, 16, 3, 1, padding=1)
        self.conv5 = nn.Conv2d(16, 16, 3, 1, padding=1)

        self.fc1 = nn.Linear(16, 3)
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.conv4.weight)
        nn.init.kaiming_normal_(self.conv5.weight)
        nn.init.kaiming_normal_(self.fc1.weight)

        if not self.use_gap:
            if self.init_maxpool == 1:
                fc0_input_size = 1120
            elif self.init_maxpool == 2:
                fc0_input_size = 240
            else:
                raise NotImplementedError()
            self.fc0 = nn.Linear(fc0_input_size, 16)
            nn.init.kaiming_normal_(self.fc0.weight)

    def forward(self, x):
        for i in range(self.init_maxpool):
            x = nn.MaxPool2d(kernel_size=2)(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = nn.MaxPool2d(kernel_size=2)(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = nn.MaxPool2d(kernel_size=2)(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = nn.MaxPool2d(kernel_size=2)(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = nn.MaxPool2d(kernel_size=2)(x)
        x = self.conv5(x)
        x = F.relu(x)

        if self.use_gap:
            x = nn.AvgPool2d(kernel_size=x.size()[2:])(x)
        x = torch.flatten(x, 1)
        if not self.use_gap:
            x = self.fc0(x)
        x = F.relu(x)
        x = self.fc1(x)
        return x

class SimpleNet2(nn.Module):
    def __init__(self):
        super(SimpleNet2, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(8, 8, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(8, 8, 3, 1, padding=1)
        self.fc1 = nn.Linear(120, 8)
        self.fc2 = nn.Linear(8, 3)
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)

    def forward(self, x):
        x = nn.MaxPool2d(kernel_size=16)(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = nn.MaxPool2d(kernel_size=2)(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = nn.MaxPool2d(kernel_size=2)(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
