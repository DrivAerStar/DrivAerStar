import os
import math
import torch
import torch.nn as nn

from einops import rearrange, repeat

from networks.common.loss_pk import  MRE

import numpy as np
import torch
from torch import nn
import lightning as L
# from .ModelBlock import TransformerBlock

import pandas as pd
from dataclasses import dataclass
from typing import Optional
import torch.nn.functional as F
from utils import register_class
from .Transolver_Irregular_Mesh import Transolver_Irregular_Mesh
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

@register_class(name=['Transolver_Irregular_Mesh'])
class Transolver_Irregular_Mesh_LightningModule(L.LightningModule):
    def __init__(self,
                 normalizer_y=None,
                 normalizer_x=None,
                 point_normalizer_y=None,
                 point_normalizer_x=None,
                 **network_args,
                 ) -> None:
        super().__init__()
        self.__name__ = 'Transolver_Irregular_Mesh'
        self.params = ModelArgs(**network_args)
        self.save_hyperparameters(ignore=['normalizer_x',
                                          'normalizer_y',
                                          'point_normalizer_y',
                                          'point_normalizer_x'])

        self.use_wandb = 1
        print('='*20, 'Parameters', '='*20)
        print(self.params)
        self.lr = self.params.lr
        print('='*50)
        if normalizer_y: self.normalizer_y = normalizer_y

        if normalizer_x:  self.normalizer_x = normalizer_x

        if point_normalizer_x:  self.point_normalizer_x = point_normalizer_x

        if point_normalizer_y:  self.point_normalizer_y = point_normalizer_y

        self.example_input_array = torch.ones([1, 10086, 7])

        self.loss = TestLoss()

        self.model = Transolver_Irregular_Mesh(
           space_dim=self.params.space_dim,
                                  n_layers=self.params.n_layers,
                                  n_hidden=self.params.n_hidden,
                                  dropout=self.params.dropout,
                                  n_head=self.params.n_heads,
                                  Time_Input=False,
                                  mlp_ratio=self.params.mlp_ratio,
                                  fun_dim=0,
                                  out_dim=self.params.out_dim,
                                  slice_num=self.params.slice_num,
                                  ref=self.params.ref,
                                  unified_pos=self.params.unified_pos)

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
        return self.model(x, None)
    
    
    
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
        drag = post_process_drag(x, y)
        drag_test = post_process_drag(x, out)
        lift = post_process_lift(x, y)
        lift_test = post_process_lift(x, out)
        output_meshs = []
        for i in range(len(vtk_paths)):
            vtk_path = vtk_paths[i]
            mesh = pv.read(vtk_path)
            
            
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

        results = {'output_meshs': output_meshs ,'vtk_paths': vtk_paths,'drag':drag, 'drag_test': drag_test,
                    'lift': lift, 'lift_test': lift_test,}
        self.test_step_outputs .append(results)
        return results
    
    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        
        drag = []
        drag_test = []
        lift = []
        lift_test = []
        ids = []
        all_output_meshs = []
        all_vtk_paths = []
        all_vtk_types = []

        for result in outputs:
            drag.append(result['drag'].item())
            lift.append(result['lift'].item())
            drag_test.append(result['drag_test'].item())
            lift_test.append(result['lift_test'].item())
            basename = os.path.basename(result['vtk_paths'][0])
            dir_of_vtk = os.path.dirname(result['vtk_paths'][0])
            type_of_vtk = os.path.basename(dir_of_vtk)
            all_vtk_types.append(type_of_vtk)
            id = basename.split('.')[0]
            ids.append(id)
            all_output_meshs.extend(result['output_meshs'])
            all_vtk_paths.extend(result['vtk_paths'])

        # drag = torch.cat(drag).cpu().numpy()
        # drag_test = torch.cat(drag_test, dim=0).cpu().numpy()
        df = pd.DataFrame({'id':ids, 'drag': drag, 'drag_test': drag_test,
                           'lift': lift, 'lift_test': lift_test,'vtk_type':all_vtk_types})
        df.to_csv(os.path.join(self.writer_path, 'drag_and_lift.csv'), index=False)


        # save all vtk files
        for i in range(len(all_output_meshs)):
            mesh_i = all_output_meshs[i]
            vtk_paths = all_vtk_paths[i]
            dir_of_vtk = os.path.dirname(vtk_paths)
            title_of_vtk = os.path.basename(dir_of_vtk)
            writer_path = self.writer_path
            if not os.path.exists(os.path.join(writer_path, 'output')):
                os.makedirs(os.path.join(writer_path, 'output')) 
            save_path = os.path.join(writer_path, 'output', title_of_vtk+'_'+os.path.basename(vtk_paths))
            mesh_i.save(save_path)
            print("save vtk file:", save_path)
       
        # return all_output_meshs, all_vtk_paths

    def predict_step(self, batch, batch_idx):
        x, _, vtk_paths = batch
        self.x_normalizer.to(x.device)
        self.y_normalizer.to(x.device)
        x_nor = self.x_normalizer.encode(x)
        out_nor = self.forward(x_nor).squeeze(-1)
        out = self.y_normalizer.decode(out_nor)
        drag_test = post_process_drag(x, out)
        lift_test = post_process_lift(x, out)
        output_meshs = []
        for i in range(len(vtk_paths)):
            vtk_path = vtk_paths[i]
            mesh = pv.read(vtk_path)

            test_pressure = out[i, :, 0].cpu().numpy()
            test_wall_ishear_stress_i = out[i, :, 1].cpu().numpy()
            test_wall_shear_stress_j = out[i, :, 2].cpu().numpy()
            test_wall_shear_stress_k = out[i, :, 3].cpu().numpy()
        
            mesh.cell_data['test_Pressure'] = test_pressure
            mesh.cell_data['test_WallShearStressi'] = test_wall_ishear_stress_i
            mesh.cell_data['test_WallShearStressj'] = test_wall_shear_stress_j
            mesh.cell_data['test_WallShearStressk'] = test_wall_shear_stress_k


            output_meshs.append(mesh)

        results = {'output_meshs': output_meshs ,'vtk_paths': vtk_paths, 'drag_test': drag_test,'lift_test': lift_test,}
        self.test_step_outputs .append(results)
        return results
    
    def on_predict_epoch_end(self):
        outputs = self.test_step_outputs

        drag = []
        drag_test = []
        lift = []
        lift_test = []
        ids = []
        all_output_meshs = []
        all_vtk_paths = []
        all_vtk_types = []

        for result in outputs:
            drag_test.append(result['drag_test'].item())
            lift_test.append(result['lift_test'].item())
            basename = os.path.basename(result['vtk_paths'][0])
            dir_of_vtk = os.path.dirname(result['vtk_paths'][0])
            type_of_vtk = os.path.basename(dir_of_vtk)
            all_vtk_types.append(type_of_vtk)
            id = basename.split('.')[0]
            ids.append(id)
            all_output_meshs.extend(result['output_meshs'])
            all_vtk_paths.extend(result['vtk_paths'])
            
            
        for i in range(len(all_output_meshs)):
            mesh_i = all_output_meshs[i]
            vtk_paths = all_vtk_paths[i]
            dir_of_vtk = os.path.dirname(vtk_paths)
            title_of_vtk = os.path.basename(dir_of_vtk)
            writer_path = self.writer_path
            if not os.path.exists(os.path.join(writer_path, 'output')):
                os.makedirs(os.path.join(writer_path, 'output')) 
            save_path = os.path.join(writer_path, 'output', title_of_vtk+'_'+os.path.basename(vtk_paths))
            mesh_i.save(save_path)
            print("save vtk file:", save_path)
       
        # return all_output_meshs, all_vtk_paths

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.params.lr, weight_decay=self.params.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.params.epochs)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',  # 
                'frequency': 1
            }
        }


    def on_before_optimizer_step(self, optimizer):
        if self.params.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.params.max_grad_norm)
