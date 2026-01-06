# DrivAerStar


Paper: 

https://drivaerstar.github.io/
https://neurips.cc/virtual/2025/loc/san-diego/poster/121752

Download:

[Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/UXVXQV)

[Hugging Face(Partial Data)](https://huggingface.co/datasets/drivaerstar/DrivAerStar-Review)

[ModelScope(Uploading)](https://www.modelscope.cn/datasets/kenalin/DrivAerPlusPlus)

Cite:
```
@article{qiu2025drivaerstar,
  title={{DrivAerStar}: An Industrial-Grade {CFD} Dataset for Vehicle Aerodynamic Optimization},
  author={Qiu, Jiyan and Lyulin Kuang and Guan Wang and Yichen Xu and Leiyao Cui and Shaotong Fu and Yixin Zhu and Rita Zhang},
  booktitle={Conference on Neural Information Processing Systems},
  year={2024}
}
```

Vedio:
<div align="center">
  <video src="https://github.com/user-attachments/assets/22440764-0d7e-438e-a44d-d0dad1006d3d" width="70%" poster="./assets/video_poster.png"> </video>
</div>


This repository contains two complementary sub-projects:

- **Production (DrivAerStar_Maker)**: Tools and scripts for generating, processing, and exporting DrivAer models (case/VTK, etc.).
- **Testing/Benchmarking (DrivAerStar_Benchmarking)**: A benchmarking suite (e.g., for the Transolver model) for deep learning training, validation, and inference based on the DrivAerStar dataset, supporting irregular mesh data and GPU-accelerated training.

Directory Structure (Example)

```
.
├── DrivAerStar_Maker/           # Production tools (STL/Mesh processing/Conversion)
├── DrivAerStar_Benchmarking/    # Benchmarking training & inference code (Models/Data modules/Configs)
└── README.md
```

---

## 1. General Environment Setup

Recommended Python version: >= 3.10. It is advised to run within a virtual or conda environment.

Install general dependencies (each sub-project might also have its own `requirements.txt`):

```bash
pip install -r requirements.txt
```

If certain dependencies cannot be installed via pip (e.g., specific PyTorch/CUDA versions, mesh libraries), please manually install the corresponding packages as indicated by the error messages or modify `requirements.txt`.

---

## 2. DrivAerStar_Maker Production Guide

**Purpose**: Use Blender and scripts to automatically generate vehicle models (STL), adjust wheel alignment, generate simulation code, and export to formats like VTK, supporting subsequent benchmarking or CFD preprocessing.

Required Software & Versions:

* Blender 4.1
* Star-CCM+ 18.06.007-R8
* pyvista 0.44.0
* Paraview 5.13.3

Recommended Directory Structure:

```
DrivAerStar_Maker/
├── 1.code_make_stls_by_blender/    # Blender Scripts: Generate STLs
├── 2.code_Wheel_Alignment/         # Wheel Alignment & Geometry Adjustment
├── 3.make_java_code_E/             # Simulation Script Code Generation
├── 4.run/                          # Run/Simulation Script Execution
└── 5.case_to_vtk/                  # Convert Simulation Case to VTK/Visualization Formats
```

Example Execution:

1. Generate STLs (Run scripts using Blender's Python CLI `blender --python` or similar):

```bash
python doe.py
python make.blender.py
```

2. Wheel Alignment (Python scripts):

```bash
python parallelru_F.py
python parallelru_N.py
python parallelru_E.py
```

3. Generate Java Code:

```bash
python make.py
```

4. Run Simulation/Processing Scripts:

```bash
./allrun_10by10.sh
```

5. Convert to VTK:

```bash
./run.sh
```

**Note**:
- You may need to specify your own paths within each script.

---

## 3. DrivAerStar_Benchmarking (Testing / Benchmarking) Guide

**Project Goal**: Train and evaluate deep learning models (example: Transolver) on irregular meshes using the DrivAerStar dataset. Uses configurable YAML files to control data loading, model parameters, and training process. Supports GPU acceleration.

Recommended Directory Structure (Example)

```
DrivAerStar_Benchmarking/
├── configs/
│   └── Transolver/
│       └── Transolver_1V_400.yaml
├── data_module/
│   └── subset/
│       └── DrivAerStar_488.py
├── networks/
│   └── Transolver/
├── main_train.py
├── main_predict.py
└── requirements.txt
```

Environment Installation

```bash
cd DrivAerStar_Benchmarking
pip install -r requirements.txt
```

Training & Inference Examples

1. Training & Validation:

```bash
python main_train.py --config-path configs/Transolver/Transolver_1V_400.yaml
```

2. Inference/Prediction:

```bash
python main_predict.py \
  --config-path configs/Transolver/Transolver_1V_400.yaml \
  --load-ckpt-path logs/checkpoints/epoch=38-step=39.ckpt
```

Example Configuration File (`Transolver_Irregular_Mesh.yaml` snippet explanation)

```yaml
Seed: 42
Data:
  args:
    batch_size: 8
    data_dir: "/dir/to/DrivAerStar/vtk/"
    val_batch_size: 1
    num_train: 50
    num_val: 10
    num_test: 10
    no_cache: False
    data_cache_file: "./dataset/cache_DrivAerStar_488.pt"
  type: DrivAerStar_488

Trainer:
  args:
    accelerator: gpu
    devices: 1
    max_epochs: 500

load_ckpt: null

network:
  args:
    lr: 0.001
    space_dim: 7
    out_dim: 4
    weight_decay: 0.0001
    n_hidden: 64
    n_layers: 4
    n_heads: 4
    max_grad_norm: 0.1
    downsample: 5
    mlp_ratio: 1
    dropout: 0.0
    ntrain: 50
    unified_pos: 0
    ref: 8
    slice_num: 16
    eval: 0
  type: Transolver_Irregular_Mesh
```







