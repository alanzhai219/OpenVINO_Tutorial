import torch
from torch import nn
import torchvision

def main():
    net = torchvision.models.alexnet(pretrained=True)
    net.eval()
    net = net.to('cpu')
    print(net)
    tmp = torch.ones(2, 3, 224, 224).to('cpu')
    out = net(tmp)
    print('alexnet out:', out.shape)
    torch.save(net, "alexnet.pth")

if __name__ == '__main__':
    main()
