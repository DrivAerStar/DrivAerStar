"""Compute ONE aerodynamic force coefficient from a VTK surface.

Reads a VTK file using PyVista, integrates pressure and wall-shear
tractions over the surface, and reports the force coefficient along ONE given
direction axis. (So: run this script twice if you want both drag and lift.)

Notes
-----
This script uses fixed VTK array keys (see constants below). If your VTK uses
different names, edit the constants (PRESSURE_KEY, AREA_KEY, NORMALS_KEY, ...).

rho can be changed with --rho (default 1.25).

Freestream speed can be set with --u (m/s).

Coefficient definition:
    C_axis = F_axis / (q_ref * A_ref)

where
    q_ref = 0.5 * rho * u^2

rho is a CLI parameter (default 1.25).

Expected data arrays (cell data preferred; point data also supported):
  - Area (cell area)
  - Normals (unit normals)
  - Pressure (scalar)
  - WallShearStressi, WallShearStressj, WallShearStressk (components)

If Area/Normals are missing, they will be computed from the geometry.
If Pressure / WSS are only present as point data, they will be converted to
cell data before integration.

Implementation note: we intentionally avoid triangulating the surface because
triangulation can duplicate per-cell arrays (like Area) onto child triangles
and over-estimate the surface integral.
"""


import argparse
import math
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pyvista as pv


# ---- VTK array keys (EDIT HERE if your file uses different names) ----
PRESSURE_KEY = "Pressure"
AREA_KEY = "Area"
NORMALS_KEY = "Normals"
WSS_I_KEY = "WallShearStressi"
WSS_J_KEY = "WallShearStressj"
WSS_K_KEY = "WallShearStressk"


def _as_vector3(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float).reshape(-1)
    if arr.size != 3:
        raise ValueError(
            f"Expected 3 values for a vector, got {arr.size}: {arr}")
    return arr


def _unit_vector(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    if not math.isfinite(n) or n <= 0.0:
        raise ValueError(f"Invalid direction vector (zero/NaN): {v}")
    return v / n


def _parse_axis(axis: str) -> np.ndarray:
    s = axis.strip().lower()
    if s in {"x", "+x"}:
        return np.array([1.0, 0.0, 0.0])
    if s == "-x":
        return np.array([-1.0, 0.0, 0.0])
    if s in {"y", "+y"}:
        return np.array([0.0, 1.0, 0.0])
    if s == "-y":
        return np.array([0.0, -1.0, 0.0])
    if s in {"z", "+z"}:
        return np.array([0.0, 0.0, 1.0])
    if s == "-z":
        return np.array([0.0, 0.0, -1.0])

    # comma/space separated vector
    for sep in [",", " ", ";", "\t"]:
        if sep in s:
            parts = [p for p in s.replace(";", ",").replace(
                "\t", ",").replace(" ", ",").split(",") if p]
            if len(parts) == 3:
                return _as_vector3(float(p) for p in parts)
    raise ValueError(
        "Invalid axis. Use x/y/z/-x/-y/-z or a 3-vector like '1,0,0' or '0 1 0'."
    )


def _ensure_surface(mesh: pv.DataSet) -> pv.PolyData:
    if isinstance(mesh, pv.PolyData):
        surf = mesh
    else:
        surf = mesh.extract_surface()
    if surf.n_cells == 0:
        raise ValueError(
            "Surface extraction produced 0 cells; check input mesh.")
    # IMPORTANT:
    # Do NOT triangulate here.
    # Many CFD exports store per-face cell-data like Area/Normals/Pressure.
    # Triangulation would split faces and may duplicate the parent cell-data to
    # each new triangle, effectively over-counting the integrated forces.
    return surf


def _ensure_cell_data(mesh: pv.PolyData, names: Iterable[str]) -> pv.PolyData:
    need_convert = any(
        (n in mesh.point_data and n not in mesh.cell_data) for n in names)
    if not need_convert:
        return mesh
    return mesh.point_data_to_cell_data(pass_point_data=True)


def _ensure_area(mesh: pv.PolyData, area_name: str = "Area") -> pv.PolyData:
    if area_name in mesh.cell_data:
        return mesh
    sized = mesh.compute_cell_sizes(length=False, area=True, volume=False)
    # PyVista uses 'Area' key for the computed areas
    if "Area" not in sized.cell_data:
        raise RuntimeError("Failed to compute cell areas.")
    if area_name != "Area":
        sized.cell_data[area_name] = sized.cell_data["Area"]
    return sized


def _ensure_normals(mesh: pv.PolyData, normals_name: str = "Normals") -> pv.PolyData:
    if normals_name in mesh.cell_data:
        return mesh
    computed = mesh.compute_normals(
        cell_normals=True,
        point_normals=False,
        auto_orient_normals=False,
        consistent_normals=False,
        inplace=False,
    )
    if "Normals" not in computed.cell_data:
        raise RuntimeError("Failed to compute cell normals.")
    if normals_name != "Normals":
        computed.cell_data[normals_name] = computed.cell_data["Normals"]
    return computed


def _build_wss_vector(mesh: pv.PolyData) -> Optional[np.ndarray]:
    # Preferred: pre-packed vector
    for candidate in ["WallShearStress", "wallShearStress", "WSS", "wss"]:
        if candidate in mesh.cell_data:
            arr = np.asarray(mesh.cell_data[candidate])
            if arr.ndim == 2 and arr.shape[1] == 3:
                return arr

    # Components (as requested)
    keys = [WSS_I_KEY, WSS_J_KEY, WSS_K_KEY]
    if all(k in mesh.cell_data for k in keys):
        wi = np.asarray(mesh.cell_data[keys[0]]).reshape(-1)
        wj = np.asarray(mesh.cell_data[keys[1]]).reshape(-1)
        wk = np.asarray(mesh.cell_data[keys[2]]).reshape(-1)
        return np.column_stack([wi, wj, wk])

    return None


@dataclass(frozen=True)
class ForceResult:
    force_total: np.ndarray
    force_pressure: np.ndarray
    force_shear: np.ndarray


def integrate_forces(
    vtk_path: str,
) -> ForceResult:
    """Integrate forces over the surface.

    Uses the traction model:
      t = -p * n + tau
    and integrates F = \int t dA (discrete sum over cells).
    """
    mesh = pv.read(vtk_path)
    surf = _ensure_surface(mesh)

    # Ensure we have required arrays on cells
    surf = _ensure_cell_data(
        surf,
        [PRESSURE_KEY, WSS_I_KEY, WSS_J_KEY, WSS_K_KEY, "WallShearStress"],
    )
    surf = _ensure_area(surf, area_name=AREA_KEY)
    surf = _ensure_normals(surf, normals_name=NORMALS_KEY)

    if PRESSURE_KEY not in surf.cell_data:
        available = sorted(list(surf.cell_data.keys()) +
                           list(surf.point_data.keys()))
        raise KeyError(
            f"Missing '{PRESSURE_KEY}' array. Available arrays: {available}")

    areas = np.asarray(surf.cell_data[AREA_KEY]).reshape(-1)
    normals = np.asarray(surf.cell_data[NORMALS_KEY])
    pressure = np.asarray(surf.cell_data[PRESSURE_KEY]).reshape(-1)

    if normals.ndim != 2 or normals.shape[1] != 3:
        raise ValueError(
            f"'{NORMALS_KEY}' must be an (N,3) array, got {normals.shape}")
    if areas.shape[0] != surf.n_cells or pressure.shape[0] != surf.n_cells or normals.shape[0] != surf.n_cells:
        raise ValueError(
            f"Cell-data size mismatch: n_cells={surf.n_cells}, "
            f"Area={areas.shape}, Normals={normals.shape}, Pressure={pressure.shape}"
        )

    # Normalize normals defensively
    nrm = np.linalg.norm(normals, axis=1)
    nrm = np.where(nrm == 0.0, 1.0, nrm)
    normals_u = normals / nrm[:, None]

    wss = _build_wss_vector(surf)
    if wss is None:
        # If WSS missing, treat shear as zero but still compute pressure force.
        wss = np.zeros((surf.n_cells, 3), dtype=float)
    else:
        wss = np.asarray(wss, dtype=float)
        if wss.shape != (surf.n_cells, 3):
            raise ValueError(
                f"WallShearStress vector must be (N,3), got {wss.shape}")

    # Per-cell force contributions
    f_pressure = (pressure[:, None] * normals_u) * areas[:, None]
    f_shear = (wss) * areas[:, None]
    f_total = f_pressure + f_shear

    return ForceResult(
        force_total=f_total.sum(axis=0),
        force_pressure=f_pressure.sum(axis=0),
        force_shear=f_shear.sum(axis=0),
    )


def compute_coeff(force: float, q: float, area_ref: float) -> float:
    if area_ref <= 0 or not math.isfinite(area_ref):
        raise ValueError(f"Invalid reference area: {area_ref}")
    if q <= 0 or not math.isfinite(q):
        raise ValueError(f"Invalid dynamic pressure q: {q}")
    return float(force) / (float(q) * float(area_ref))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute ONE force coefficient along ONE axis by integrating Pressure and WallShearStress on a VTK surface.",
    )
    parser.add_argument("--vtk", required=True,
                        help="Path to VTK/VTU/VTP file")
    parser.add_argument("--area-ref", type=float, required=True,
                        help="Reference (projected) area A_ref")
    parser.add_argument(
        "--axis",
        type=str,
        required=True,
        help="Force direction axis: x/y/z/-x/-y/-z or vector like '1,0,0'",
    )

    parser.add_argument(
        "--component",
        choices=["total", "pressure", "shear"],
        default="total",
        help="Which contribution to report (default: total).",
    )

    parser.add_argument(
        "--rho",
        type=float,
        default=1.25,
        help="Density rho used for q_ref=0.5*rho*u^2 (default: 1.25)",
    )

    parser.add_argument(
        "--u",
        type=float,
        default=40.0,
        help="Freestream speed u used for q_ref=0.5*rho*u^2 (default: 40.0)",
    )

    args = parser.parse_args()

    axis_dir = _unit_vector(_parse_axis(args.axis))

    rho = float(args.rho)
    u = float(args.u)
    q = 0.5 * rho * (u ** 2)

    res = integrate_forces(vtk_path=args.vtk)

    f_total = res.force_total
    f_p = res.force_pressure
    f_w = res.force_shear

    if args.component == "pressure":
        f_vec = f_p
    elif args.component == "shear":
        f_vec = f_w
    else:
        f_vec = f_total

    force_along_axis = float(np.dot(f_vec, axis_dir))
    coeff = compute_coeff(force_along_axis, q=q, area_ref=args.area_ref)

    # Also compute breakdown along axis for convenience
    force_p = float(np.dot(f_p, axis_dir))
    force_w = float(np.dot(f_w, axis_dir))
    coeff_p = compute_coeff(force_p, q=q, area_ref=args.area_ref)
    coeff_w = compute_coeff(force_w, q=q, area_ref=args.area_ref)

    np.set_printoptions(precision=6, suppress=True)
    print("=== Integrated forces (global) ===")
    print(f"F_total   = {f_total}")
    print(f"F_pressure= {f_p}")
    print(f"F_shear   = {f_w}")
    print("=== Direction ===")
    print(f"axis_dir = {axis_dir}")
    print("=== Along-axis result ===")
    print(f"component = {args.component}")
    print(f"F_axis(total)   = {float(np.dot(f_total, axis_dir)):.6g}")
    print(f"F_axis(pressure)= {force_p:.6g}")
    print(f"F_axis(shear)   = {force_w:.6g}")
    print("=== Coefficient ===")
    print(
        f"A_ref = {args.area_ref:.6g}, rho = {rho:.6g}, u = {u:.6g}, q_ref = {q:.6g}")
    print(f"C_axis({args.component}) = {coeff:.8f}")
    print(
        f"C_axis(total)    = {compute_coeff(float(np.dot(f_total, axis_dir)), q=q, area_ref=args.area_ref):.8f}")
    print(f"C_axis(pressure) = {coeff_p:.8f}")
    print(f"C_axis(shear)    = {coeff_w:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
