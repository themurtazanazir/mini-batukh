import torch
from torch import nn
from torch.nn import functional as F, Conv2d
from torchvision.models import resnet50



class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class ImgEncoder(nn.Module):
    def __init__(self):
        super(ImgEncoder, self).__init__()
        self.base_model = resnet50(pretrained=True, progress=True)
        ## delete the last avgpool and fc layer
        self.base_model.conv1 = Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.base_model.maxpool = Identity()
        self.base_model.avgpool = Identity()
        self.base_model.fc = Identity()

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)
        x = torch.flatten(x, 2).transpose(1, 2)
        x = F.log_softmax(x, dim=-1)
        return x

    def forward(self, x):
        return self._forward_impl(x)





if __name__ == '__main__':
    m = ImgEncoder()
    inp = torch.rand(size = (1, 3, 32, 800))
    with torch.no_grad():
        print(m(inp).shape)