import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transform as transforms

mb_size = 64
transform = transforms.Compose((transforms.ToTensor()))
