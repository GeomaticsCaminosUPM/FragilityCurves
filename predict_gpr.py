"""
predict_gpr.py
==============
Inferencia con el modelo GPR+PCA entrenado.
Aplica automáticamente la corrección conservadora si el modelo la tiene.

Uso:
    python predict_gpr.py \
        --exp_name exp01_baseline \
        --floors 2 --position C --diaphragm R \
        --direction Y --asym_x 1 --asym_y 1 \
        --agg "3R11_2R11_1R11"
"""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset_processor import build_feature_vector, parse_unit


def parse_aggregate(agg_str):
    tokens = agg_str.split("_")
    return [parse_unit(t) for t in tokens[:3]]


def predict_curve(bundle, feat_vec, conservative=True):
    gprs       = bundle["gprs"]
    pca        = bundle["pca"]
    scaler     = bundle["scaler"]
    correction = bundle.get("correction", np.zeros(pca.n_components_))
    n_comp     = bundle["n_components"]

    feat_sc = scaler.transform(feat_vec.reshape(1, -1))
    comps   = np.array([[g.predict(feat_sc)[0] for g in gprs]])  # (1, n_comp)
    y_pred  = pca.inverse_transform(comps)[0]                    # (n_points,)

    if conservative:
        y_pred = y_pred - correction

    return y_pred


def main(args):
    exp_dir = Path("./experiments") / args.exp_name
    model_path = exp_dir / "gpr_model.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en {model_path}\n"
            f"Entrena primero con: python gpr_model.py --exp_name {args.exp_name}")

    bundle = joblib.load(model_path)
    x_axis = bundle["x_axis"]

    # Cargar métricas del experimento para mostrarlas
    results_path = exp_dir / "results.json"
    exp_info = {}
    if results_path.exists():
        with open(results_path) as f:
            exp_info = json.load(f)

    # Construir feature vector
    agg_units = parse_aggregate(args.agg)
    info = {
        "floors":      args.floors,
        "position":    args.position,
        "diaphragm":   args.diaphragm,
        "direction":   args.direction,
        "asym_x":      args.asym_x,
        "asym_y":      args.asym_y,
        "agg_units":   agg_units,
        "is_isolated": agg_units[0] is None,
    }
    feat_vec = build_feature_vector(info)

    # Predicción base y conservadora
    y_base = predict_curve(bundle, feat_vec, conservative=False)
    y_cons = predict_curve(bundle, feat_vec, conservative=True)

    print(f"\nExperimento : {args.exp_name}")
    if exp_info:
        mb = exp_info.get("metrics_no_bias", {})
        mc = exp_info.get("metrics_conserv", {})
        print(f"LOO R²      : {mb.get('r2', '?'):.4f}  "
              f"(conserv: {mc.get('r2', '?'):.4f})")
        print(f"LOO below%  : {mb.get('pct_below', '?'):.1f}%  "
              f"(conserv: {mc.get('pct_below', '?'):.1f}%)")

    print(f"\nCurva base      → min={y_base.min():.4f}  max={y_base.max():.4f}")
    print(f"Curva conserv.  → min={y_cons.min():.4f}  max={y_cons.max():.4f}")
    print(f"Corrección media aplicada: {(y_base - y_cons).mean():.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(x_axis, y_base, lw=2, color="orange", ls="--", label="GPR (sin corrección)")
    ax.plot(x_axis, y_cons, lw=2, color="green",  label="GPR conservador")
    ax.fill_between(x_axis, y_cons, y_base, alpha=0.15, color="orange",
                    label="Margen conservador")

    title = (f"{args.floors}F · {args.position} · Diaph:{args.diaphragm} · "
             f"Dir:{args.direction} · AsymX:{args.asym_x} · AsymY:{args.asym_y}\n"
             f"Agregado: {args.agg}  |  exp: {args.exp_name}")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()

    out_png = exp_dir / (f"pred_{args.floors}{args.position}{args.diaphragm}"
                         f"{args.direction}_ax{args.asym_x}_ay{args.asym_y}.png")
    plt.savefig(out_png, dpi=120)
    if args.plot:
        plt.show()
    plt.close()
    print(f"\nPlot guardado en {out_png}")

    # Guardar CSV
    out_csv = out_png.with_suffix(".csv")
    with open(out_csv, "w") as f:
        f.write("x,y_base,y_conservative\n")
        for xi, yb, yc in zip(x_axis, y_base, y_cons):
            f.write(f"{xi:.6f},{yb:.6f},{yc:.6f}\n")
    print(f"CSV guardado en {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name",  required=True,
                        help="Nombre del experimento (carpeta en ./experiments/)")
    parser.add_argument("--floors",    type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--position",  required=True, choices=["I", "L", "C", "R"])
    parser.add_argument("--diaphragm", required=True, choices=["R", "F"])
    parser.add_argument("--direction", required=True, choices=["X", "Y"])
    parser.add_argument("--asym_x",    type=int, required=True)
    parser.add_argument("--asym_y",    type=int, required=True)
    parser.add_argument("--agg",       default="ISOL_ISOL_ISOL")
    parser.add_argument("--plot",      action="store_true")
    args = parser.parse_args()
    main(args)