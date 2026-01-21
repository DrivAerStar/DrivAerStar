# Changelog

## 2026-01-21

### Added

- New force-coefficient utility folder: `DrivAerStar_Maker/6.vtk_to_force_coefficient/`
  - `force_coefficient.py`: computes **one** force coefficient along a specified axis by integrating surface `Pressure` and `WallShearStress*` from a VTK file (PyVista + NumPy).
  - `frontal_area/`: added frontal-area CSV tables (`frontal_area_E.csv`, `frontal_area_F.csv`, `frontal_area_N.csv`).

- Added design parameter tables under `DrivAerStar_Maker/1.code_make_stls_by_blender/lhs_parameters_example/`:
  - `lhs_parameters_Estate_v3.csv`
  - `lhs_parameters_Fast_v3.csv`
  - `lhs_parameters_Notch_v3.csv`

### Changed

- Updated `README.md` to document the new force-coefficient workflow and example command.
