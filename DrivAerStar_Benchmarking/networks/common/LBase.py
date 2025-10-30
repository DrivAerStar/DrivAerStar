
import torch
import torch.nn as nn
import lightning as L

from networks.common.loss import Rating

class TransformerEncoderModel(L.LightningModule):
    def __init__(self, normalizer_y = None,normalizer_x = None
                 ):
        super().__init__()
        
        if normalizer_y:
            self.normalizer_y = normalizer_y
            
        if normalizer_x:
            self.normalizer_x = normalizer_x
        
        # Loss function
        self.loss = nn.MSELoss()
        self.rating = Rating()
        self.save_hyperparameters()

    def forward(self, feature ):
        return feature

    def training_step(self, batch, batch_idx):
        feature, label = batch
        feature_nor = self.normalizer_x.encode(feature)        
        label_nor = self.normalizer_y.encode(label)       
         
        z = self(feature_nor)
        
        loss = self.loss(z.flatten(), label_nor.flatten())
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss


    def validation_step(self, batch, batch_idx):
        feature, label = batch
        feature_nor = self.normalizer_x.encode(feature)        
        label_nor = self.normalizer_y.encode(label)   
        z = self(feature_nor)
        
        loss = self.loss(z.flatten(), label_nor.flatten())
        
        predict_y = self.normalizer_y.encode(z)
        rating = self.rating(label.flatten(), predict_y.flatten())
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_rating', rating, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return {'val_loss': loss,'val_rating': rating,}

    def predict_step(self, batch, batch_idx):
        return self(batch)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer
