import torch
from torch import nn
from torch.nn import functional as F, Conv2d
from torchvision.models import resnet50
import pytorch_lightning as pl
from src.utils import GreedyCTCDecoder


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ImgEncoder(nn.Module):
    def __init__(self, vocab_size):
        super(ImgEncoder, self).__init__()
        self.base_model = resnet50(pretrained=True, progress=True)
        # delete the last avgpool and fc layer
        self.base_model.conv1 = Conv2d(3, 64, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.base_model.maxpool = Identity()
        self.base_model.avgpool = Identity()
        self.base_model.fc = Identity()
        self.final_layer = Conv2d(2048, vocab_size, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1), bias=False)

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
        x = self.final_layer(x)
        x = torch.flatten(x, 2).transpose(1, 2)
        x = F.log_softmax(x, dim=-1)
        return x

    def forward(self, x):
        return self._forward_impl(x)


class FinalModel(pl.LightningModule):

    def __init__(self, vocab_size):
        super(FinalModel, self).__init__()
        self.img_encoder = ImgEncoder(vocab_size)

    def forward(self, x):
        return self.img_encoder(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
        return [optimizer], []

    def training_step(self, batch, batch_idx):
        x, y, target_len = batch
        y = torch.flatten(y)
        y_hat = self(x)
        y_hat = torch.transpose(y_hat, 0, 1)
        # print(y_hat.shape)
        # !! Remove hard coded value later
        input_len = torch.full(
            size=(y_hat.shape[1],), fill_value=200, dtype=torch.long)
        loss = F.ctc_loss(y_hat, y, input_lengths=input_len,
                          target_lengths=target_len, reduction='mean')
        # print(loss)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, target_len = batch
        y = torch.flatten(y)
        y_hat = self(x)
        y_hat = torch.transpose(y_hat, 0, 1)

        # !! Remove hard coded value later
        input_len = torch.full(
            size=(y_hat.shape[1],), fill_value=200, dtype=torch.long)
        val_loss = F.ctc_loss(
            y_hat, y, input_lengths=input_len, target_lengths=target_len)

        self.log("val_loss", val_loss)

    def predict_step(self, batch, batch_idx):

        x, y, target_len = batch
        print(y)
        pred = self(x)
        return pred


if __name__ == '__main__':
    m = ImgEncoder(vocab_size=21)
    inp = torch.rand(size=(1, 3, 32, 400))
    with torch.no_grad():
        print(m(inp).shape)
