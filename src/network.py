from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchinfo import summary

from config import device

class ResidualBlock(nn.Module): 
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(output_dim)
        self.relu = nn.ReLU()
        self.identity = nn.Identity()
    
    def forward(self, x: torch.Tensor): 
        residual = self.identity(x)
        
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        out = x + residual
        out = self.relu(out)
        
        return out

class PolicyValueNet(nn.Module): 
    def __init__(self) -> None:
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16), 
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32), 
            nn.ReLU(),
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            ResidualBlock(32, 32)
        )

        self.policy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 3 * 3, 9)
        )
        
        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 3 * 3, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: 
        _feature = self.backbone(x)
        policy = self.policy_head(_feature)
        value = self.value_head(_feature)
        return policy, value

if __name__ == "__main__":
    model = PolicyValueNet()
    summary(model, (1, 1, 3, 3), device=device)
