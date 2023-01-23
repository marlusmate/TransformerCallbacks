from lightningmodule.vit_tiny import VisionTransformer
import os
from torch import optim, nn, utils, Tensor
import pytorch_lightning as pl
from lr_scheduler import build_scheduler
from model_optimizer import build_adamw, set_weight_decay

class vit_encoder(pl.LightningModule):
    
    def __init__(self, ):
        super(vit_encoder, self).__init__()
        self.feature_extractor = VisionTransformer(pre_logits=True, weight_init='')
        self.classifier = nn.Linear(192,3)
        self.loss = nn.CrossEntropyLoss()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3)
        
        return optimizer

    def forward(self, x):
        output = self.feature_extractor(x)
        pred = nn.functional.softmax(self.classifier(output))
        return pred

    def training_step(self, batch, batch_idx):
        x, pv, y= batch
        pred = self.forward(x)
        loss = self.loss(pred, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, val_batch_idx):
        x, pv, y = val_batch
        pred = self.forward(x)
        val_loss = self.loss(pred, y)
        self.log("val_loss", val_loss)

    def one_batch(self):
        return
    
    """
    def backward(self, trainer, loss, optimizer, optimizer_idx):
        loss.backward()
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.step()
    """