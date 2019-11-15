import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

mb_size = 64
transform = transforms.Compose([transforms.ToTensor()])
trainset =torchvision.datasets.MNIST(root='./NewData', download=True,train=True,transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=mb_size)

data_iter = iter(trainloader)

images, labels = data_iter.next()
test = images.view(images.size(0),-1)
print(images.size())

Z_dim = 100
X_dim = test.size(1)

def imshow(img):
  im = torchvision.utils.make_grid(img)
  npimg = im.numpy()
  print(npimg.shape)
  plt.figure(figsize=(8,8))
  plt.imshow(np.transpose(npimg,(1,2,0)))
  plt.xticks([]),plt.yticks([])
  plt.show()

imshow(images)

def init_weight(m):
  if type(m) == nn.Linear:
    nn.init.xavier_uniform_(m.weight)
    m.bias.data.fill_(0)

class Gen(nn.Module):
  def __init__(self):
    super(Gen, self).__init__()
    self.model = nn.Sequential(
        nn.Linear(Z_dim,h_dim),
        nn.ReLU(),
        nn.Linear(h_dim,X_dim),
        nn.Sigmoid(),
    )
    self.model.apply(init_weight)
  def forword(self, input):
    return self.model(input)

class Dis(nn.Module):
  def __init__(self):
    super(Dis,self).__init__()
    self.model = nn.Sequential(
        nn.Linear(X_dim,h_dim),
        nn.ReLU(),
        nn.Linear(h_dim,1),
        nn.Sigmoid(),
    )
    self.model.apply(init_weight)
  def forword(self, input):
    return self.model(input)

# Z_dim = 100
Z_dim = 28*28
X_dim = test.size(1)
h_dim = 128
lr = 1e-3

G = Gen()
D = Dis()
G_solver = optim.Adam(G.parameters(),lr = lr)
D_solver = optim.Adam(D.parameters(),lr = lr)
