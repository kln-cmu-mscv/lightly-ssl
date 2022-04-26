import torch
import torch.nn as nn
import pytorch_lightning as pl
import lightly
from lightly.models.modules.heads import SimCLRProjectionHead
from lightly.loss import NTXentLoss
from network.resnet_3d import RESNET18

class SimCLR_R3D_Model(pl.LightningModule):
    def __init__(self, r3d : RESNET18, max_epcohs, **kwargs):
        super().__init__()

        # create a ResNet3d backbone and remove the classification head
        self.backbone = r3d
        hidden_dim = r3d.resnet.fc.out_features

        # create projection head
        self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, 128)

        self.max_epochs = max_epcohs
        self.criterion = NTXentLoss()

        self.lr_base = kwargs['lr_base']
        self.batch_size = kwargs['batch_size']
        self.lr_steps = kwargs['lr_steps']
        self.lr_factor = kwargs['lr_factor']
        self.fine_tune = kwargs['fine_tune']

        return

    def forward(self, x):
        h = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(h)
        return z

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        # optim = torch.optim.SGD(
        #     self.parameters(), lr=6e-3, momentum=0.9, weight_decay=5e-4, nesterov = True
        # )
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optim, self.max_epochs
        # )

        lr_base = self.lr_base
        fine_tune = self.fine_tune

        param_base_layers, param_new_layers = self.discriminative_lr(self.backbone, fine_tune)
        optim = torch.optim.SGD([{'params': param_base_layers, 'lr_mult': 0.2},  
                                 {'params': param_new_layers, 'lr_mult': 1.0},
                                 {'params': self.projection_head.parameters(), 'lr_mult': 1.0}],
                                lr=lr_base, 
                                momentum=0.9, 
                                weight_decay=0.0001, 
                                nesterov=True)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, self.max_epochs
        )

        return [optim], [scheduler]

    def discriminative_lr(self, model, fine_tune = True):
        # config optimization
        param_base_layers = []
        param_new_layers = []
        name_base_layers = []
        for name, param in model.named_parameters():
            if fine_tune:
                if ('projection_head' in name) or ('fc' in name):
                    param_new_layers.append(param)
                else:
                    param_base_layers.append(param)
                    name_base_layers.append(name)
            else:
                param_new_layers.append(param)

        return param_base_layers, param_new_layers