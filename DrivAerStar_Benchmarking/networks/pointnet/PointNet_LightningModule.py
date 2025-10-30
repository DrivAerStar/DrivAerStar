import os
import math
import torch
import torch.nn as nn
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from einops import rearrange, repeat

from networks.common.loss_pk import  MRE

import numpy as np
import torch
from torch import nn
import lightning as L
# from .ModelBlock import TransformerBlock


from dataclasses import dataclass
from typing import Optional
import torch.nn.functional as F
from utils import register_class

from ..common.Transovler_testloss import TestLoss
import pyvista as pv

@dataclass
class ModelArgs:
    lr: float = 1e-3
    space_dim: int =3
    out_dim: int = 4
    epochs: int =500
    weight_decay:float =1e-5
    n_hidden:int =128
    n_layers:int =8
    n_heads:int =8
    max_grad_norm:float =0.1
    downsample:int =5
    mlp_ratio:int =1
    dropout: float=0.0
    ntrain:int =1000
    unified_pos:int =0
    ref:int =8
    slice_num:int =34
    eval:int =0

def post_process_drag(x, y):
    """
    post_process_CD
    :param x: [B, N, 7]  posision[x,y,z]  arae[1]  normal[x,y,z]
    :param y: [B, N, 4]  pressure[1]  wall shear stress[x,y,z]
    :return:
    """
    press = y[:, :, 0]
    arae = x[:, :, 3]
    normal = x[:, :, 4]
    wss_x = y[:, :, 1]
    drag_press = torch.sum(press * arae * normal, dim=1)
    drag_wss = torch.sum(wss_x*arae , dim=1)
    drag = drag_press + drag_wss
    return drag

def post_process_lift(x, y):
    """
    post_process_CD
    :param x: [B, N, 7]  posision[x,y,z]  arae[1]  normal[x,y,z]
    :param y: [B, N, 4]  pressure[1]  wall shear stress[x,y,z]
    :return:
    """
    press = y[:, :, 0]
    arae = x[:, :, 3]
    normal = x[:, :, 6]
    wss_x = y[:, :, 3]
    lift_press = torch.sum(press * arae * normal, dim=1)
    lift_wss = torch.sum(wss_x*arae , dim=1)
    lift = lift_press + lift_wss
    return lift

class TNet(nn.Module):
    """"""
    def __init__(self, k=7):
        super().__init__()
        self.k = k
        self.conv_layers = nn.Sequential(
            nn.Conv1d(k, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, k*k)
        )

    def forward(self, x):
        batch_size = x.size(0)
        
        # 
        x = self.conv_layers(x)          # [B, 256, N]
        x = torch.max(x, 2, keepdim=True)[0]  # [B, 256, 1]
        x = x.view(batch_size, -1)       # [B, 256]
        
        # 
        x = self.fc_layers(x)            # [B, k*k]
        x = x.view(batch_size, self.k, self.k)  # [B, k, k]
        
        # 
        identity = torch.eye(self.k, device=x.device).unsqueeze(0).repeat(batch_size, 1, 1)
        return x + identity

class PointNet(nn.Module):
    """PointNet"""
    def __init__(self, in_channels=7, out_channels=4, feature_transform=False):
        super().__init__()
        self.feature_transform = feature_transform
        
        # 
        self.input_tnet = TNet(k=in_channels)
        
        # 
        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        
        # 
        if feature_transform:
            self.feature_tnet = TNet(k=64)
        
        # 
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.bn3 = nn.BatchNorm1d(256)
        
        # 
        self.residual = nn.Sequential(
            nn.Conv1d(64, 256, 1),  # conv3
            nn.BatchNorm1d(256)
        )
        
        # 
        self.pointwise_layers = nn.Sequential(
            nn.Conv1d(256, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, out_channels, 1)
        )

    def forward(self, x):
        # : [B, N, 7] ->  [B, 7, N]
        x = x.transpose(2, 1)
        
        # 
        trans_input = self.input_tnet(x)          # [B, 7, 7]
        x = torch.bmm(trans_input, x)             # [B, 7, N]
        
        # 
        x = F.relu(self.bn1(self.conv1(x)))       # [B, 64, N]
        features = x.clone()  # 
        
        # 
        if self.feature_transform:
            trans_feat = self.feature_tnet(x)     # [B, 64, 64]
            x = torch.bmm(trans_feat, x)          # [B, 64, N]
        
        # 
        x = F.relu(self.bn2(self.conv2(x)))       # [B, 128, N]
        x = self.bn3(self.conv3(x))               # [B, 256, N]
        
        # 
        residual = self.residual(features)        # [B, 256, N]
        x = x + residual                          # [B, 256, N]
        
        # 
        x = self.pointwise_layers(x)              # [B, 4, N]
        
        #  [B, N, 4]
        return x.transpose(2, 1)
       
@register_class('pointnet')
class PointNet_LightningModule(L.LightningModule):
    def __init__(self, 
                 normalizer_y=None,
                 normalizer_x=None,
                 point_normalizer_y=None,
                 point_normalizer_x=None,
                 **network_args):
        super().__init__()
        self.__name__ = 'PointNet_Irregular_Mesh'
        self.params = ModelArgs(**network_args)
        
        # TransformerPointNet
        self.model = PointNet(in_channels=7, out_channels=4, feature_transform=True)
        
        # ...
        self.loss = TestLoss()
        
        # 
        if normalizer_y: self.normalizer_y = normalizer_y
        if normalizer_x: self.normalizer_x = normalizer_x
        if point_normalizer_x: self.point_normalizer_x = point_normalizer_x
        if point_normalizer_y: self.point_normalizer_y = point_normalizer_y

    def forward(self, x):
        #  [B, 1, N, 4] -> [B, N, 4]
        return self.model(x)
    
    def set_normalizer(self,normalizer_x,normalizer_y):
        self.x_normalizer = normalizer_x
        self.x_normalizer.cuda()
        self.y_normalizer = normalizer_y
        self.y_normalizer.cuda()
    
    
    def set_test_args(self, writer_path):
        self.writer_path = writer_path
        self.test_step_outputs = []
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        self.x_normalizer.to(x.device)
        self.y_normalizer.to(x.device)
        x_nor = self.x_normalizer.encode(x)
        out_nor = self.forward(x_nor).squeeze(-1)
        out = self.y_normalizer.decode(out_nor)
        loss = self.loss(out, y)
        with torch.no_grad():
            out_press = out[:, :, 0]
            out_wss = out[:, :, 1:4]
            y_press = y[:, :, 0]
            y_wss = y[:, :, 1:4]
            loss_press = self.loss(out_press, y_press)
            loss_wss = self.loss(out_wss, y_wss)
            drag = post_process_drag(x, y)
            drag_pred = post_process_drag(x, out)
            lift = post_process_lift(x, y)
            lift_pred = post_process_lift(x, out)
            drag_loss = self.loss(drag_pred, drag)
            lift_loss = self.loss(lift_pred, lift)
        self.log('train_loss_drag', drag_loss, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)
        self.log('train_loss_lift', lift_loss, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)
        self.log('train_loss_press', loss_press, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)
        self.log('train_loss_wss', loss_wss, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)
        self.log('train_loss', loss, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)
        
        return loss


    def forward(self, x):
        return self.model(x)
    
    
    
    def validation_step(self, batch, batch_idx):
        
        x, y = batch
        self.x_normalizer.to(x.device)
        self.y_normalizer.to(x.device)
        x_nor = self.x_normalizer.encode(x)
        out_nor = self.forward(x_nor).squeeze(-1)
        out = self.y_normalizer.decode(out_nor)
        loss = self.loss(out, y)
        out_press = out[:, :, 0]
        out_wss = out[:, :, 1:4]
        y_press = y[:, :, 0]
        y_wss = y[:, :, 1:4]
        loss_press = self.loss(out_press, y_press)
        loss_wss = self.loss(out_wss, y_wss)
        drag = post_process_drag(x, y)
        drag_pred = post_process_drag(x, out)
        lift = post_process_lift(x, y)
        lift_pred = post_process_lift(x, out)
        drag_loss = self.loss(drag_pred, drag)
        lift_loss = self.loss(lift_pred, lift)
        self.log('val_loss_drag', drag_loss, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)
        self.log('val_loss_lift', lift_loss, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)
        self.log('val_loss_press', loss_press, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)
        self.log('val_loss_wss', loss_wss, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)
        

        results = {'val_loss': loss}

        return results

    def test_step(self, batch, batch_idx):
        x, y, vtk_paths = batch
        self.x_normalizer.to(x.device)
        self.y_normalizer.to(x.device)
        x_nor = self.x_normalizer.encode(x)
        out_nor = self.forward(x_nor).squeeze(-1)
        out = self.y_normalizer.decode(out_nor)
        loss = self.loss(out, y)

        # vtk
        output_meshs = []
        for i in range(len(vtk_paths)):
            vtk_path = vtk_paths[i]
            mesh = pv.read(vtk_path)
            # print("read vtk file:", vtk_path)
            cell_pressure = mesh.cell_data.get('Pressure').flatten()
            cell_wall_ishear_stress_i = mesh.cell_data.get('WallShearStressi').flatten()
            cell_wall_shear_stress_j = mesh.cell_data.get('WallShearStressj').flatten()
            cell_wall_shear_stress_k = mesh.cell_data.get('WallShearStressk').flatten()

            test_pressure = out[i, :, 0].cpu().numpy()
            test_wall_ishear_stress_i = out[i, :, 1].cpu().numpy()
            test_wall_shear_stress_j = out[i, :, 2].cpu().numpy()
            test_wall_shear_stress_k = out[i, :, 3].cpu().numpy()
            
            diff_pressure = test_pressure - cell_pressure
            diff_wall_ishear_stress_i = test_wall_ishear_stress_i - cell_wall_ishear_stress_i
            diff_wall_shear_stress_j = test_wall_shear_stress_j - cell_wall_shear_stress_j
            diff_wall_shear_stress_k = test_wall_shear_stress_k - cell_wall_shear_stress_k
        
            mesh.cell_data['test_Pressure'] = test_pressure
            mesh.cell_data['test_WallShearStressi'] = test_wall_ishear_stress_i
            mesh.cell_data['test_WallShearStressj'] = test_wall_shear_stress_j
            mesh.cell_data['test_WallShearStressk'] = test_wall_shear_stress_k
            mesh.cell_data['diff_Pressure'] = diff_pressure
            mesh.cell_data['diff_WallShearStressi'] = diff_wall_ishear_stress_i
            mesh.cell_data['diff_WallShearStressj'] = diff_wall_shear_stress_j
            mesh.cell_data['diff_WallShearStressk'] = diff_wall_shear_stress_k

            output_meshs.append(mesh)

        results = {'output_meshs': output_meshs ,'vtk_paths': vtk_paths}
        self.test_step_outputs .append(results)
        return results
    
    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        all_output_meshs = []
        all_vtk_paths = []

        # output_meshsvtk_paths
        for result in outputs:
            all_output_meshs.extend(result['output_meshs'])
            all_vtk_paths.extend(result['vtk_paths'])

          # vtk
        for i in range(len(all_output_meshs)):
            mesh_i = all_output_meshs[i]
            vtk_paths = all_vtk_paths[i]
            dir_of_vtk = os.path.dirname(vtk_paths)
            writer_path = self.writer_path
            if not os.path.exists(os.path.join(writer_path, 'output')):
                os.makedirs(os.path.join(writer_path, 'output')) 
            mesh_i.save(os.path.join(writer_path, 'output', dir_of_vtk+'_'+os.path.basename(vtk_paths)))
            print("save vtk file:", os.path.join(writer_path, 'output', os.path.basename(vtk_paths)))
       
       
        # return all_output_meshs, all_vtk_paths

    def predict_step(self, batch, batch_idx):
        pass
    
    def predict_epoch_end(self, outputs):
        preds = torch.cat(outputs)
        print("Prediction Results DataFrame:\n", preds.mean())

    def configure_optimizers(self):
        # 
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.params.lr, weight_decay=self.params.weight_decay)

        # 
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.params.epochs)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',  # epoch
                'frequency': 1
            }
        }


    def on_before_optimizer_step(self, optimizer):
        # 
        if self.params.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.params.max_grad_norm)
