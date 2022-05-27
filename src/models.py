import math
from unicodedata import bidirectional
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F, Conv2d
from torchvision.models import resnet50, resnet18
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
        self.final_layer = Conv2d(2048, 512, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        self.linear1 = nn.Linear(2048, 512)
        
        self.rnn1 = nn.GRU(512, 256, batch_first=True, bidirectional=True, num_layers=2)
        # self.rnn2 = nn.GRU(256, 256, batch_first=True, bidirectional=True)
        self.linear2 = nn.Linear(256*2, vocab_size)
         

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        # print(x.shape)
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)
        # print(x.shape)
        x = self.final_layer(x)
        x = x.permute(0, 3, 1, 2) # [batch_size, width, channels, height]
        # print(x.shape)

        batch_size = x.size(0)
        T = x.size(1)
        x = x.view(batch_size, T, -1) 
        # print(x.shape)
        # x = torch.flatten(x, 2).transpose(1, 2)
        # print(x.shape)
        # print(x.shape)
        # h = torch.zeros(4, 1, 21, dtype=torch.float32, device=x.device)
        x = self.linear1(x)

        x, h = self.rnn1(x)
        # feature_size = x.size(2)
        # x = x[:, :, :feature_size//2] + x[:, :, feature_size//2:]
        # x, h = self.rnn2(x)
        x = self.linear2(x)
        # x, _ = self.rnn2(x)
        x = F.log_softmax(x, dim=-1)
        return x

    def forward(self, x):
        return self._forward_impl(x)



class CRNN(nn.Module):
    
    def __init__(self, num_chars, rnn_hidden_size=256, dropout=0.1):
        
        super(CRNN, self).__init__()
        self.num_chars = num_chars
        self.rnn_hidden_size = rnn_hidden_size
        self.dropout = dropout
        
        # CNN Part 1
        resnet_modules = list(resnet18(pretrained=True, progress=True).children())[:-3]
        self.cnn_p1 = nn.Sequential(*resnet_modules)
        
        # CNN Part 2
        self.cnn_p2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3,6), stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.linear1 = nn.Linear(512, 256)
        
        # RNN
        self.rnn1 = nn.GRU(input_size=rnn_hidden_size, 
                            hidden_size=rnn_hidden_size,
                            bidirectional=True, 
                            batch_first=True)
        self.rnn2 = nn.GRU(input_size=rnn_hidden_size, 
                            hidden_size=rnn_hidden_size,
                            bidirectional=True, 
                            batch_first=True)
        self.linear2 = nn.Linear(self.rnn_hidden_size*2, num_chars)
        
        
    def forward(self, batch):
        # print(batch.shape)
        batch = self.cnn_p1(batch)
        # print(batch.shape)
        # print(batch.size()) # torch.Size([-1, 256, 4, 13])
        
        batch = self.cnn_p2(batch) # [batch_size, channels, height, width]
        # print(batch.shape)
        # print(batch.size())# torch.Size([-1, 256, 4, 10])
        
        batch = batch.permute(0, 3, 1, 2) # [batch_size, width, channels, height]
        # print(batch.shape)
        # print(batch.size()) # torch.Size([-1, 10, 256, 4])
         
        batch_size = batch.size(0)
        T = batch.size(1)
        batch = batch.view(batch_size, T, -1) # [batch_size, T==width, num_features==channels*height]
        # print(batch.shape)
        # print(batch.size()) # torch.Size([-1, 10, 1024])
        
        batch = self.linear1(batch)
        # print(batch.size()) # torch.Size([-1, 10, 256])
        
        batch, hidden = self.rnn1(batch)
        feature_size = batch.size(2)
        batch = batch[:, :, :feature_size//2] + batch[:, :, feature_size//2:]
        # print(batch.size()) # torch.Size([-1, 10, 256])
        
        batch, hidden = self.rnn2(batch)
        # print(batch.size()) # torch.Size([-1, 10, 512])
        
        batch = self.linear2(batch)
        # print(batch.size()) # torch.Size([-1, 10, 20])
        
        batch = batch.permute(1, 0, 2) # [T==10, batch_size, num_classes==num_features]
        # print(batch.size()) # torch.Size([10, -1, 20])
        
        return F.log_softmax(batch, 2)

class CustomCTCLoss(torch.nn.Module):
    # T x B x H => Softmax on dimension 2
    def __init__(self, dim=2):
        super().__init__()
        self.dim = dim
        self.ctc_loss = torch.nn.CTCLoss(reduction='mean', zero_infinity=True)

    def forward(self, logits, labels,
            prediction_sizes, target_sizes):
        EPS = 1e-7
        loss = self.ctc_loss(logits, labels, prediction_sizes, target_sizes)
        loss = self.sanitize(loss)
        return self.debug(loss, logits, labels, prediction_sizes, target_sizes)
    
    def sanitize(self, loss):
        EPS = 1e-7
        if abs(loss.item() - float('inf')) < EPS:
            return torch.zeros_like(loss)
        if math.isnan(loss.item()):
            return torch.zeros_like(loss)
        return loss

    def debug(self, loss, logits, labels,
            prediction_sizes, target_sizes):
        if math.isnan(loss.item()):
            print("Loss:", loss)
            print("logits:", logits)
            print("labels:", labels)
            print("prediction_sizes:", prediction_sizes)
            print("target_sizes:", target_sizes)
            raise Exception("NaN loss obtained. But why?")
        return loss


class FinalModel(pl.LightningModule):

    def __init__(self, vocab_size):
        super(FinalModel, self).__init__()
        # self.img_encoder = ImgEncoder(vocab_size)
        self.img_encoder = CRNN(vocab_size)
        self.ctc = CustomCTCLoss()

    def forward(self, x):
        return self.img_encoder(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3) # 5e-4 for v_255 restarting
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=5, factor=0.5)
        # return [optimizer], [lr_scheduler]
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": lr_scheduler,
            "monitor": "train_loss",
            # "frequency": "indicates how often the metric is updated"
            # If "monitor" references validation metrics, then "frequency" should be set to a
            # multiple of "trainer.check_val_every_n_epoch".
        },
    }

    def training_step(self, batch, batch_idx):
        x, y, target_len = batch
        y = torch.flatten(y)
        y_hat = self(x)
        # y_hat = y_hat.permute(1, 0, 2)
        # print(y_hat.shape)
        # !! Remove hard coded value later
        input_len = torch.full(
            size=(y_hat.shape[1],), fill_value=22, dtype=torch.long)
        # loss = F.ctc_loss(y_hat, y, input_lengths=input_len,
        #                   target_lengths=target_len, reduction='mean', zero_infinity=True)
        loss = self.ctc(y_hat, y, input_len, target_len)
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
            size=(y_hat.shape[1],), fill_value=50, dtype=torch.long)
        val_loss = F.ctc_loss(
            y_hat, y, input_lengths=input_len, target_lengths=target_len)

        self.log("val_loss", val_loss)

    def predict_step(self, batch, batch_idx):
        x, y, target_len = batch
        # print(y)
        pred = self(x)
        return pred




if __name__ == '__main__':
    m = FinalModel(vocab_size=21)
    inp = torch.rand(size=(1, 3, 32, 400))
    # with torch.no_grad():
    o = m(inp)
    # o = m(inp)
    print(o.shape)
