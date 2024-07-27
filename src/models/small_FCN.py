import torch.nn as nn
import torch.nn.functional as F

class small_FCN(nn.Module):
    def __init__(self):
        super(small_FCN, self).__init__()
        
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Linear(512, 230)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)
        
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2)
        
        x = F.relu(self.conv3(x))
        x = F.max_pool1d(x, 2)
        
        x = F.relu(self.conv4(x))
        
        x = self.gap(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.fc(x)
        
        return x