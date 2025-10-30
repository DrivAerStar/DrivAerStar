import os
import numpy as np
from pathlib import Path
from typing import *
from torch.nn.utils.rnn import pad_sequence

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from utils import register_class
import torch.nn as nn
import pyvista as pv

from einops import rearrange


class MSNormalizer(nn.Module):
    def __init__(self, mean, std, eps=1e-07):
        super(MSNormalizer, self).__init__()
        self.mean = mean
        self.std = std
        self.eps = torch.tensor(eps)

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x):
        x = x * (self.std + self.eps) + self.mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()
        self.eps = self.eps.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()
        self.eps = self.eps.cpu()

    def to(self,device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        self.eps = self.eps.to(device)


class CustomDataset1(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return feature, label

class CustomDataset_test(Dataset):
    def __init__(self, features, labels, vtk_path):
        self.features = features
        self.labels = labels
        self.vtk_path = vtk_path

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        vtk_path = self.vtk_path[idx]
        return feature, label , vtk_path


def read_vtk(file_path: str):
    try:
        mesh = pv.read(file_path)
        cell_centers = mesh.cell_centers().points
        cell_areas = mesh.cell_data.get('Area')
        cell_normals = mesh.cell_data.get('Normals')
        cell_pressure = mesh.cell_data.get('Pressure')
        cell_wall_ishear_stress_i = mesh.cell_data.get('WallShearStressi')
        cell_wall_shear_stress_j = mesh.cell_data.get('WallShearStressj')
        cell_wall_shear_stress_k = mesh.cell_data.get('WallShearStressk')

        return {
            'cell_centers': cell_centers,
            'cell_areas': cell_areas,
            'cell_normals': cell_normals,
            'cell_pressure': cell_pressure,
            'cell_wall_ishear_stress_i': cell_wall_ishear_stress_i,
            'cell_wall_shear_stress_j': cell_wall_shear_stress_j,
            'cell_wall_shear_stress_k': cell_wall_shear_stress_k
        }
    except FileNotFoundError:
        print(f":  {file_path} 。")
        return None
    except Exception as e:
        print(f": : {e}")
        return None


@register_class(name=['DrivAerStar_1Vehicle'])
class DrivAerStar_1Vehicle(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str = "./dataset/",
                 batch_size: int = 8,
                 val_batch_size: int = 1,
                 num_train: int = 50,
                 num_val: int = 10,
                 num_test: int = 10,
                 no_cache=False , 
                 data_cache_file : str = "./dataset/cache_DrivAerStar_1Vehicle.pt"):

        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        self.val_batch_size = val_batch_size
        self.no_cache = no_cache
        self.data_cache_file = data_cache_file

    def setup_normalizer(self):
  
        self.data_normalizer_x = MSNormalizer(
            mean=self.feature_norm1,
            std=self.feature_norm2,
        )

        self.data_normalizer_y = MSNormalizer(
            mean=self.label_norm1,
            std=self.label_norm2
        )
        
        self.normalizer_x = self.data_normalizer_x
        self.normalizer_y = self.data_normalizer_y


    def setup_data(self):
        self.inputs = []
        self.outputs = []
        self.using_vtk = []
        for file in self.vtk_files:
            data = read_vtk(file)
            if data is not None:
                print(f"Processing file: {file}")
                try:
                    input_data = np.hstack([
                        data['cell_centers'], # N,3
                        data['cell_areas'].reshape(-1, 1), # N,1
                        data['cell_normals'] # N,3
                    ])
                    output_data = np.hstack([
                        data['cell_pressure'].reshape(-1, 1),
                        data['cell_wall_ishear_stress_i'].reshape(-1, 1),
                        data['cell_wall_shear_stress_j'].reshape(-1, 1),
                        data['cell_wall_shear_stress_k'].reshape(-1, 1)
                    ])

                    #  input_data  NaN 
                    if np.isnan(input_data).any():
                        print(f"Warning: Input data in {file} contains NaN values. Skipping this file.")
                        continue

                    #  output_data  NaN 
                    if np.isnan(output_data).any():
                        print(f"Warning: Output data in {file} contains NaN values. Skipping this file.")
                        continue

                    #  output_data  20000(Pa)
                    cell_pressure = data['cell_pressure']
                    if np.any(np.abs(cell_pressure) > 20000):
                        print(f"Warning: 'cell_pressure' data in {file} has absolute values higher than 20000. Skipping this file.")
                        continue
                    
                    self.inputs.append(torch.tensor(input_data, dtype=torch.float32))
                    self.outputs.append(torch.tensor(output_data, dtype=torch.float32))
                    self.using_vtk.append(file)
                except Exception as e:
                    print(f"Error processing file {file}: {e}")
                
        print("Dataloading is over.")


        self.feature_norm1 = [torch.mean(input_i, dim=0) for input_i in self.inputs]
        self.feature_norm2 = [torch.std(input_i, dim=0) for input_i in self.inputs]
        
        self.feature_norm1 = torch.stack(self.feature_norm1, dim=0).mean(dim=0)
        self.feature_norm2 = torch.stack(self.feature_norm2, dim=0).mean(dim=0)
        
        self.label_norm1 = [torch.mean(output_i, dim=0) for output_i in self.outputs]
        self.label_norm2 = [torch.std(output_i, dim=0) for output_i in self.outputs]
        
        self.label_norm1 = torch.stack(self.label_norm1, dim=0).mean(dim=0)
        self.label_norm2 = torch.stack(self.label_norm2, dim=0).mean(dim=0)
            
    def setup(self, stage: Optional[str] = None):
        self.files = os.listdir(self.data_dir)
        self.files.sort()
        self.vtk_files = []
        for file in self.files:
            if file.endswith('.vtk'):
                self.vtk_files.append(os.path.join(self.data_dir, file))

        if self.no_cache:
            self.setup_data()
        else:
            if self.data_cache_file is not None and os.path.exists(self.data_cache_file):
                data = torch.load(self.data_cache_file)
                self.inputs = data['inputs']
                self.outputs = data['outputs']
                self.feature_norm1 = data['feature_norm1']
                self.feature_norm2 = data['feature_norm2']
                self.label_norm1 = data['label_norm1']
                self.label_norm2 = data['label_norm2']
                self.using_vtk = data['using_vtk']
            else:
                self.setup_data()
                save_dir = os.path.dirname(self.data_cache_file)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir,exist_ok=True)
                if self.data_cache_file is not None:
                    torch.save({
                        'inputs': self.inputs,
                        'outputs': self.outputs,
                        'feature_norm1': self.feature_norm1,
                        'feature_norm2': self.feature_norm2,
                        'label_norm1': self.label_norm1,
                        'label_norm2': self.label_norm2,
                        'using_vtk': self.using_vtk,
                    }, self.data_cache_file)
            
        # 
        self.setup_normalizer()

        all_data = list(zip(self.inputs, self.outputs, self.using_vtk))
        assert len(all_data) > self.num_train + self.num_val + self.num_test, "，、。"
        train_end_index = self.num_train
        val_start_index = len(all_data) - self.num_val - self.num_test
        val_end_index = len(all_data) - self.num_test

        train_data = all_data[:train_end_index]
        val_data = all_data[val_start_index:val_end_index]
        test_data = all_data[val_end_index:]

        train_inputs, train_outputs, train_vtk_files = zip(*train_data)
        val_inputs, val_outputs, val_vtk_files  = zip(*val_data)
        test_inputs, test_outputs, test_vtk_files  = zip(*test_data)

        # 
        train_dataset = CustomDataset1(train_inputs, train_outputs)
        val_dataset = CustomDataset1(val_inputs, val_outputs)
        test_dataset = CustomDataset_test(test_inputs, test_outputs,test_vtk_files)
        def collate_fn(data):
            inputs_list = [x for x, _ in data]
            targets_list = [y for _, y in data]
            inputs = pad_sequence(inputs_list).permute(1, 0, 2)  # B, T1, F
            targets = pad_sequence(targets_list).permute(1, 0, 2)  # B, T1, F
            return (inputs, targets)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
        self.val_loader = DataLoader(val_dataset, batch_size=self.val_batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader
    
    def normalizer(self):
        return self.normalizer_x, self.normalizer_y
    
    def teardown(self, stage: Optional[str] = None):
        pass

    
