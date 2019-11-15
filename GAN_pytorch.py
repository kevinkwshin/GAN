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
# trainset =torchvision.datasets.MNIST(root='./NewData', download=True,train=True,transform=transform)
trainset =torchvision.datasets.(root='./NewData', download=True,train=True,transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=mb_size)

data_iter = iter(trainloader)
images, labels = data_iter.next()

test = images.view(images.size(0),-1)
print(images.size())

Z_dim = 100
H_dim = 128
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
class Gen(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(Z_dim, H_dim),
            nn.ReLU(),
            nn.Linear(H_dim, X_dim),
            nn.Sigmoid()
        )
          
    def forward(self, input):
        return self.model(input)

class Dis(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(X_dim, H_dim),
            nn.ReLU(),
            nn.Linear(H_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, input):
        return self.model(input)

Z_dim = 100
X_dim = test.size(1)
h_dim = 128
lr = 1e-3

device = 'cuda'
G = Gen().to(device)
D = Dis().to(device)

for epoch in range(20):
  G_loss_run = 0.0
  D_loss_run = 0.0
  for i, data in enumerate(trainloader):
    X, _ = data
    mb_size = X.size(0)
    X = X.view(X.size(0),-1).to(device)

    one_labels = torch.ones(mb_size,1).to(device)
    zero_labels = torch.zeros(mb_size,1).to(device)
    
    z = torch.randn(mb_size, Z_dim).to(device)
    
    D_real = D(X)
    D_fake = D(G(z))
    
    D_real_loss = F.binary_cross_entropy(D_real, one_labels)
    D_fake_loss = F.binary_cross_entropy(D_fake, zero_labels)
    D_loss = D_real_loss + D_fake_loss

    D_solver.zero_grad()
    D_loss.backward()
    D_solver.step()

    z = torch.randn(mb_size,Z_dim).to(device)

    D_fake = D(G(z))
    G_loss = F.binary_cross_entropy(D_fake, one_labels)

    G_solver.zero_grad()
    G_loss.backward()
    G_solver.step()
    
    G_loss_run += G_loss.item()
    D_loss_run += D_loss.item()

  print('Epoch: {}, G_loss: {}, D_loss: {}'.format(epoch, G_loss_run/(i+1), D_loss_run/(i+1)))
    
  samples = G(z).detach()
  samples = samples.view(mb_size,1,28,28).cpu()
  imshow(samples)
G_solver = optim.Adam(G.parameters(),lr = lr)
D_solver = optim.Adam(D.parameters(),lr = lr)


