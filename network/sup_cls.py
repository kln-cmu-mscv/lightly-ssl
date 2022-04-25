import torch
import torch.nn as nn
import pytorch_lightning as pl
import lightly
from lightly.models.modules.heads import SimCLRProjectionHead
from lightly.loss import NTXentLoss
from network.resnet_3d import RESNET18
from network.lr_scheduler import MultiFactorScheduler
from torchmetrics import Accuracy

class SupCls_R3D_Model(pl.LightningModule):
    def __init__(self, r3d : RESNET18, max_epochs, **kwargs):
        super().__init__()

        # create a ResNet3d backbone and remove the classification head
        self.backbone = r3d
        hidden_dim = r3d.resnet.fc.out_features

        # create projection head
        self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, 11)

        self.max_epochs = max_epochs
        self.criterion = nn.CrossEntropyLoss()
        
        self.lr_base = kwargs['lr_base']
        self.batch_size = kwargs['batch_size']
        self.lr_steps = kwargs['lr_steps']
        self.lr_factor = kwargs['lr_factor']
        self.fine_tune = kwargs['fine_tune']

        # Metrics
        self.accuracy1 = Accuracy(top_k = 1)
        self.accuracy2 = Accuracy(top_k = 2)

        return

    def forward(self, x):
        h = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(h)
        return z

    def training_step(self, batch, batch_idx):
        x0, labels, _ = batch
        z0 = self.forward(x0)
        loss = self.criterion(z0, labels)
    
        self.accuracy1(nn.functional.softmax(z0, dim = -1).detach().cpu(), labels.cpu())
        self.accuracy2(nn.functional.softmax(z0, dim = -1).detach().cpu(), labels.cpu())

        self.log("train_loss_sup", loss)
        self.log("train_acc_step top_1: ", self.accuracy1)
        self.log("train_acc_step top_2: ", self.accuracy2)
        return loss
    
    def training_epoch_end(self, outs):
        # log epoch metric
        self.log("train_acc_epoch top_1: ", self.accuracy1)
        self.log("train_acc_epoch top_2: ", self.accuracy2)

    def configure_optimizers(self):
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

        # scheduler = MultiFactorScheduler(base_lr=lr_base, 
        #                                  steps=[int(x/(batch_size*num_worker)) for x in lr_steps],
        #                                  factor=lr_factor, 
        #                                  step_counter=0)
	
        # optim = torch.optim.SGD(
        #     self.parameters(), lr=6e-2, momentum=0.9, weight_decay=5e-4
        # )
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

        