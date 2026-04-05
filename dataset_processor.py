"""
dataset_processor.py
====================
Parsea los archivos .txt del dataset de dwellings, extrae los inputs
desde el nombre del archivo, interpola todas las curvas a una longitud
fija y exporta tensores PyTorch listos para entrenar.

Uso:
    python dataset_processor.py --data_dir . --output_dir ./processed --n_points 256
    python dataset_processor.py --data_dir . --output_dir ./processed --n_points 128

Cada ejecución genera un archivo versionado (dataset_p256.pt, dataset_p128.pt...)
y actualiza ./processed/datasets.json con el registro de versiones disponibles.

Para usar una versión concreta en un experimento GPR:
    python gpr_model.py --exp_name exp01 --dataset ./processed/dataset_p256.pt
"""

import os
import re
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from scipy.interpolate import interp1d
from typing import Optional

# ─────────────────────────────────────────────
# 1.  PARSING DEL NOMBRE DE ARCHIVO
# ─────────────────────────────────────────────

DWELLING_RE = re.compile(
    r"^(\d+)"
    r"-([ILCR])"
    r"-([RF])"
    r"-([XY])"
    r"-(\d+)"
    r"-(\d+)"
    r"-(.+)$"
)

UNIT_RE = re.compile(r"(\d+)([RF])(\d)(\d)")


def parse_unit(token: str):
    if token == "ISOL":
        return None
    m = UNIT_RE.match(token)
    if not m:
        raise ValueError(f"Token no reconocido: {token}")
    return {
        "floors":    int(m.group(1)),
        "diaphragm": m.group(2),
        "asym_x":    int(m.group(3)),
        "asym_y":    int(m.group(4)),
    }


def parse_filename(stem: str) -> Optional[dict]:
    m = DWELLING_RE.match(stem)
    if not m:
        return None
    floors, position, diaphragm, direction, asym_x, asym_y, agg_str = m.groups()
    units = [parse_unit(tok) for tok in agg_str.split("_")]
    return {
        "floors":      int(floors),
        "position":    position,
        "diaphragm":   diaphragm,
        "direction":   direction,
        "asym_x":      int(asym_x),
        "asym_y":      int(asym_y),
        "agg_units":   units,
        "is_isolated": units[0] is None,
    }


# ─────────────────────────────────────────────
# 2.  CARGA DE CURVA
# ─────────────────────────────────────────────

def load_curve(filepath: Path):
    data = np.loadtxt(filepath)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    x, y = data[:, 0], data[:, 1]
    order = np.argsort(x)
    return x[order], y[order]


# ─────────────────────────────────────────────
# 3.  INTERPOLACION A LONGITUD FIJA
# ─────────────────────────────────────────────

def interpolate_curve(x, y, n_points=256, x_min=None, x_max=None):
    if x_min is None: x_min = x.min()
    if x_max is None: x_max = x.max()
    x_new = np.linspace(x_min, x_max, n_points)
    fn = interp1d(x, y, kind="linear", bounds_error=False, fill_value=(y[0], y[-1]))
    return x_new, fn(x_new).astype(np.float32)


# ─────────────────────────────────────────────
# 4.  FEATURE ENCODING
# ─────────────────────────────────────────────

def encode_position(pos: str) -> list:
    mapping = {"I": 0, "L": 1, "C": 2, "R": 3}
    v = [0.0] * 4
    v[mapping[pos]] = 1.0
    return v


def encode_unit(unit: Optional[dict]) -> list:
    if unit is None:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    return [
        unit["floors"] / 3.0,
        1.0 if unit["diaphragm"] == "R" else 0.0,
        1.0 if unit["diaphragm"] == "F" else 0.0,
        (unit["asym_x"] - 1) / 4.0,
        (unit["asym_y"] - 1) / 4.0,
        0.0,
    ]


def build_feature_vector(info: dict) -> np.ndarray:
    """
    Vector de features (27 dims):
        position one-hot  (4)
        diaphragm R/F     (1)
        direction X/Y     (1)
        floors norm       (1)
        asym_x norm       (1)
        asym_y norm       (1)
        unit_L encoding   (6)
        unit_C encoding   (6)
        unit_R encoding   (6)
    """
    pos_oh = encode_position(info["position"])
    diaph  = [0.0 if info["diaphragm"] == "R" else 1.0]
    direc  = [0.0 if info["direction"]  == "X" else 1.0]
    floors = [info["floors"] / 3.0]
    asym   = [(info["asym_x"] - 1) / 4.0,
              (info["asym_y"] - 1) / 4.0]
    units  = info["agg_units"]
    u0 = encode_unit(units[0] if len(units) > 0 else None)
    u1 = encode_unit(units[1] if len(units) > 1 else None)
    u2 = encode_unit(units[2] if len(units) > 2 else None)
    feat = pos_oh + diaph + direc + floors + asym + u0 + u1 + u2
    return np.array(feat, dtype=np.float32)


FEATURE_NAMES = (
    ["pos_I", "pos_L", "pos_C", "pos_R",
     "diaphragm", "direction",
     "floors_norm", "asym_x_norm", "asym_y_norm"] +
    [f"u{i}_{k}" for i in range(3)
     for k in ["floors", "is_R", "is_F", "asym_x", "asym_y", "is_isol"]]
)


# ─────────────────────────────────────────────
# 5.  PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

def build_dataset(data_dir: str, n_points: int = 256, plot: bool = False) -> dict:
    data_dir  = Path(data_dir)
    txt_files = sorted(data_dir.rglob("*.txt"))
    txt_files = [f for f in txt_files if not f.name.startswith("indicaciones")]
    print(f"Archivos encontrados: {len(txt_files)}")

    x_min_global, x_max_global = np.inf, -np.inf
    raw_data = []

    for fpath in txt_files:
        info = parse_filename(fpath.stem)
        if info is None:
            print(f"  [SKIP] {fpath.name}")
            continue
        try:
            x, y = load_curve(fpath)
        except Exception as e:
            print(f"  [ERROR] {fpath.name}: {e}")
            continue
        x_min_global = min(x_min_global, x.min())
        x_max_global = max(x_max_global, x.max())
        raw_data.append((info, x, y, fpath.name))

    print(f"  Rango X global  : [{x_min_global:.4f}, {x_max_global:.4f}]")
    print(f"  Muestras validas: {len(raw_data)}")

    features_list, curves_list, names_list = [], [], []
    x_axis = np.linspace(x_min_global, x_max_global, n_points)

    for info, x, y, fname in raw_data:
        _, y_interp = interpolate_curve(x, y, n_points, x_min_global, x_max_global)
        features_list.append(build_feature_vector(info))
        curves_list.append(y_interp)
        names_list.append(fname)

    X = torch.tensor(np.stack(features_list), dtype=torch.float32)
    Y = torch.tensor(np.stack(curves_list),   dtype=torch.float32)

    print(f"\n  Tensor features : {X.shape}  (N x F)")
    print(f"  Tensor curvas   : {Y.shape}  (N x P)")

    dataset = {
        "X": X, "Y": Y,
        "x_axis":        torch.tensor(x_axis, dtype=torch.float32),
        "feature_names": FEATURE_NAMES,
        "filenames":     names_list,
        "x_min":         x_min_global,
        "x_max":         x_max_global,
        "n_points":      n_points,
    }

    N       = len(X)
    idx     = torch.randperm(N)
    n_train = int(0.70 * N)
    n_val   = int(0.15 * N)
    dataset["idx_train"] = idx[:n_train]
    dataset["idx_val"]   = idx[n_train:n_train + n_val]
    dataset["idx_test"]  = idx[n_train + n_val:]
    print(f"  Splits -> train: {n_train}  val: {n_val}  test: {N-n_train-n_val}")

    if plot:
        _plot_sample(x_axis, Y.numpy(), names_list)

    return dataset


# ─────────────────────────────────────────────
# 6.  PLOT DE MUESTRA
# ─────────────────────────────────────────────

def _plot_sample(x_axis, Y, names, n=6):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib no disponible.")
        return
    idx = np.random.choice(len(Y), min(n, len(Y)), replace=False)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for i, ax in enumerate(axes.flatten()):
        if i >= len(idx):
            ax.axis("off"); continue
        ax.plot(x_axis, Y[idx[i]], lw=1.5)
        ax.set_title(names[idx[i]][:40], fontsize=7)
        ax.set_xlabel("X"); ax.set_ylabel("Y")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("sample_curves.png", dpi=120)
    plt.show()


# ─────────────────────────────────────────────
# 7.  ENTRY POINT
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   required=True)
    parser.add_argument("--output_dir", default="./processed")
    parser.add_argument("--n_points",   type=int, default=256,
                        help="Numero de puntos de interpolacion. "
                             "Genera dataset_p<n>.pt (ej: dataset_p256.pt)")
    parser.add_argument("--plot",       action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = Path(args.output_dir)

    dataset = build_dataset(args.data_dir, args.n_points, args.plot)

    version      = f"p{args.n_points}"
    dataset_file = f"dataset_{version}.pt"

    torch.save(dataset,      out_path / dataset_file)
    torch.save(dataset["X"], out_path / f"X_features_{version}.pt")
    torch.save(dataset["Y"], out_path / f"Y_curves_{version}.pt")

    # Registro acumulativo de versiones
    registry_path = out_path / "datasets.json"
    registry = {}
    if registry_path.exists():
        with open(registry_path) as f:
            registry = json.load(f)

    registry[version] = {
        "file":          dataset_file,
        "n_points":      args.n_points,
        "n_features":    int(dataset["X"].shape[1]),
        "n_samples":     int(dataset["X"].shape[0]),
        "x_min":         float(dataset["x_min"]),
        "x_max":         float(dataset["x_max"]),
        "feature_names": FEATURE_NAMES,
    }

    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)

    print(f"\n Dataset guardado en {out_path}/")
    print(f"  {dataset_file}")
    print(f"\nVersiones disponibles (datasets.json):")
    for v, info in registry.items():
        marker = " <-- nuevo" if v == version else ""
        print(f"  {v:<10} {info['file']:<30} "
              f"({info['n_points']} pts, {info['n_samples']} muestras){marker}")
    print(f"\nUsar en experimento GPR:")
    print(f"  python gpr_model.py --exp_name mi_exp --dataset {out_path / dataset_file}")


if __name__ == "__main__":
    main()