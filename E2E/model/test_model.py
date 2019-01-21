import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, noClasses, channels=3):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(channels, 4, kernel_size=5, padding=(2, 2))
        self.conv2 = nn.Conv2d(4, 6, kernel_size=5, padding=(2, 2))
        self.conv2_bn1 = nn.BatchNorm2d(6)
        self.conv3 = nn.Conv2d(6, 8, kernel_size=5, padding=(2, 2))
        self.conv2_bn2 = nn.BatchNorm2d(8)
        self.conv4 = nn.Conv2d(8, 10, kernel_size=5, padding=(2, 2))
        self.conv2_bn3 = nn.BatchNorm2d(10)
        self.conv5 = nn.Conv2d(10, 12, kernel_size=5, padding=(2, 2))
        self.conv5_bn3 = nn.BatchNorm2d(12)
        self.conv5_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(48, 100)
        self.fc = nn.Linear(100, noClasses)
        self.featureSize = 48

    def forward(self, x, feature=False, T=1, labels=False, scale=None, predictClass=False):

        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.conv2_bn1(self.conv2(x))
        x = F.relu(F.max_pool2d(self.conv2_bn2(self.conv3(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2_bn3(self.conv4(x)), 2))
        x = F.relu(F.max_pool2d(self.conv5_drop(self.conv5_bn3(self.conv5(x))), 2))
        x = x.view(x.size(0), -1)
        if feature:
            return x / torch.norm(x, 2, 1).unsqueeze(1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)

        if labels:
            if predictClass:
                return F.softmax(self.fc(x) / T), F.softmax(self.fc2(x) / T)
            return F.softmax(self.fc(x) / T)

        if scale is not None:
            x = self.fc(x)
            temp = F.softmax(x / T)
            temp = temp * scale
            return temp

        if predictClass:
            return F.log_softmax(self.fc(x) / T), F.log_softmax(self.fc2(x) / T)
        return F.log_softmax(self.fc(x) / T)
