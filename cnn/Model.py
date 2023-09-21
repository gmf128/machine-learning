import torch
from torch import nn
from torch.nn import functional as F

# 定义模型
class myCNN(nn.Module):
    dim_hidden = []
    def __init__(self):
        super(myCNN, self).__init__()
        """定义网络结果"""
        # dim_hidden : 定义网络中间层参数shape
        dim_hidden = [32, 128, 512, 1024, 256]
        self.dim_hidden = dim_hidden
        """卷积层"""
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=dim_hidden[0], kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=dim_hidden[0], out_channels=dim_hidden[1], kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=dim_hidden[1], out_channels=dim_hidden[2], kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(dim_hidden[0])
        self.bn2 = nn.BatchNorm2d(dim_hidden[1])
        self.bn3 = nn.BatchNorm2d(dim_hidden[2])
        """全连接层"""
        self.fc1 = nn.Linear(in_features=dim_hidden[2] * 7 * 7, out_features=dim_hidden[3])  # 注意输入的dim：28/2/2 = 7
        self.fc2 = nn.Linear(in_features=dim_hidden[3], out_features=dim_hidden[4])
        self.fc3 = nn.Linear(in_features=dim_hidden[4], out_features=10)

    def forward(self, x):
        """forward前向运算"""
        x = F.relu(self.bn1(self.conv1(x)))
        x = nn.functional.max_pool2d(x, kernel_size=2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = nn.functional.max_pool2d(x, kernel_size=2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(-1, self.dim_hidden[2] * 7 * 7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    """backward自动计算"""

