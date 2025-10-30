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

from .DrivAerStar_1Vehicle import MSNormalizer, CustomDataset1, CustomDataset_test, read_vtk


@register_class(name=['DrivAerStar_3Vehicles'])
class DrivAerStar_3Vehicles(pl.LightningDataModule):
    def __init__(self,
                 data_dirs: list = ["./dataset/vtk_E", "./dataset/vtk_F", "./dataset/vtk_N"],
                 batch_size: int = 8,
                 val_batch_size: int = 1,
                 num_train: int = 50,
                 num_val: int = 10,
                 num_test: int = 10,
                 no_cache=False,
                 data_cache_file: str = "./dataset/cache_DrivAerStar_3Vehicles.pt"):

        super().__init__()
        self.data_dirs = data_dirs
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

    def setup(self, stage: Optional[str] = None):
        assert len(self.data_dirs) == 3, "need 3 dirs" 

        
        for data_dir in self.data_dirs:
            if not os.path.exists(data_dir):
                
                raise FileNotFoundError(f"directory {data_dir} does not exist.")

        if not self.no_cache and os.path.exists(self.data_cache_file):
            data = torch.load(self.data_cache_file)
            self.inputs = data['inputs']
            self.outputs = data['outputs']
            self.feature_norm1 = data['feature_norm1']
            self.feature_norm2 = data['feature_norm2']
            self.label_norm1 = data['label_norm1']
            self.label_norm2 = data['label_norm2']
            self.using_vtk = data['using_vtk']
            self.train_files = data['train_files']
            self.val_files = data['val_files']
            self.test_files = data['test_files']
        else:
            
            self.vtk_files_3v = []
            for data_dir in self.data_dirs:
                files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.vtk')])
                if not files:
                    raise FileNotFoundError("directory {} has no vtk files.".format(data_dir))
                    
                self.vtk_files_3v.append(files)

            
            total_train = self.num_train
            total_val = self.num_val
            total_test = self.num_test

            
            train_per_vehicle = total_train // 3
            remainder_train = total_train % 3
            train_counts = [train_per_vehicle + 1 if i < remainder_train else train_per_vehicle for i in range(3)]

            
            val_per_vehicle = total_val // 3
            remainder_val = total_val % 3
            val_counts = [val_per_vehicle + 1 if i < remainder_val else val_per_vehicle for i in range(3)]

            
            test_per_vehicle = total_test // 3
            remainder_test = total_test % 3
            test_counts = [test_per_vehicle + 1 if i < remainder_test else test_per_vehicle for i in range(3)]

            
            for i in range(3):
                available = len(self.vtk_files_3v[i])
                required = train_counts[i] + val_counts[i] + test_counts[i]
                if available < required:
                    raise ValueError(f"{i+1} need {required} but only {available} available.")

            
            self.train_files = []
            self.val_files = []
            self.test_files = []
            for i in range(3):
                files = self.vtk_files_3v[i]
                
                train = files[:train_counts[i]]
                remaining = files[train_counts[i]:]
                
                val = remaining[:val_counts[i]]
                
                test = remaining[val_counts[i]:val_counts[i] + test_counts[i]]
                self.train_files.extend(train)
                self.val_files.extend(val)
                self.test_files.extend(test)

            
            def load_data(files):
                inputs = []
                outputs = []
                using_vtk = []
                for file in files:
                    data = read_vtk(file)
                    if data is None:
                        continue
                    try:
                        input_data = np.hstack([
                            data['cell_centers'],
                            data['cell_areas'].reshape(-1, 1),
                            data['cell_normals']
                        ])
                        output_data = np.hstack([
                            data['cell_pressure'].reshape(-1, 1),
                            data['cell_wall_ishear_stress_i'].reshape(-1, 1),
                            data['cell_wall_shear_stress_j'].reshape(-1, 1),
                            data['cell_wall_shear_stress_k'].reshape(-1, 1)
                        ])
                        if np.isnan(input_data).any() or np.isnan(output_data).any():
                            print(f" {file}  have NaN")
                            continue
                        if np.any(np.abs(data['cell_pressure']) > 20000):
                            print(f" {file}  have NaN")
                            continue
                        inputs.append(torch.tensor(input_data, dtype=torch.float32))
                        outputs.append(torch.tensor(output_data, dtype=torch.float32))
                        using_vtk.append(file)
                    except Exception as e:
                        
                        print(" skip a file due to error in processing.")
                return inputs, outputs, using_vtk

            
            train_inputs, train_outputs, train_using_vtk = load_data(self.train_files)
            if not train_inputs:
                raise RuntimeError("no valid training data found.")

            
            all_train_inputs = torch.cat(train_inputs, dim=0)
            self.feature_norm1 = all_train_inputs.mean(dim=0)
            self.feature_norm2 = all_train_inputs.std(dim=0)

            all_train_outputs = torch.cat(train_outputs, dim=0)
            self.label_norm1 = all_train_outputs.mean(dim=0)
            self.label_norm2 = all_train_outputs.std(dim=0)

            
            all_files = self.train_files + self.val_files + self.test_files
            all_inputs, all_outputs, all_using_vtk = load_data(all_files)
            self.inputs = all_inputs
            self.outputs = all_outputs
            self.using_vtk = all_using_vtk

            
            if self.data_cache_file is not None:
                save_dir = os.path.dirname(self.data_cache_file)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                torch.save({
                    'inputs': self.inputs,
                    'outputs': self.outputs,
                    'feature_norm1': self.feature_norm1,
                    'feature_norm2': self.feature_norm2,
                    'label_norm1': self.label_norm1,
                    'label_norm2': self.label_norm2,
                    'using_vtk': self.using_vtk,
                    'train_files': self.train_files,
                    'val_files': self.val_files,
                    'test_files': self.test_files,
                }, self.data_cache_file)

        
        self.setup_normalizer()

        
        file_to_index = {file: idx for idx, file in enumerate(self.using_vtk)}

        
        train_indices = [file_to_index[file] for file in self.train_files if file in file_to_index]
        val_indices = [file_to_index[file] for file in self.val_files if file in file_to_index]
        test_indices = [file_to_index[file] for file in self.test_files if file in file_to_index]

        
        train_dataset = CustomDataset1(
            [self.inputs[i] for i in train_indices],
            [self.outputs[i] for i in train_indices],
            
            
        )
        val_dataset = CustomDataset1(
            [self.inputs[i] for i in val_indices],
            [self.outputs[i] for i in val_indices],
            
            
        )
        test_dataset = CustomDataset_test(
            [self.inputs[i] for i in test_indices],
            [self.outputs[i] for i in test_indices],
            [self.using_vtk[i] for i in test_indices],
            
            
        )

        
        def collate_fn(batch):
            inputs_list = [x for x, y in batch]
            targets_list = [y for x, y in batch]
            inputs = pad_sequence(inputs_list, batch_first=True, padding_value=0)
            targets = pad_sequence(targets_list, batch_first=True, padding_value=0)
            return inputs, targets

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda batch: (  
                pad_sequence([x for x, y, _ in batch], batch_first=True, padding_value=0),
                pad_sequence([y for x, y, _ in batch], batch_first=True, padding_value=0),
                [f for _, _, f in batch]
            )
        )

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
