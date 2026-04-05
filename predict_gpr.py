"""
predict_gpr.py
==============
Inferencia con el modelo GPR+PCA entrenado.
Aplica corrección conservadora e incertidumbre (±1σ, ±2σ).

Uso:
    python predict_gpr.py \
        --exp_name exp01_baseline \
        --floors 2 --position C --diaphragm R \
        --direction Y --asym_x 1 --asym_y 1 \
        --agg "3R11_2R11_1R11" --plot

Salida en ./experiments/<exp_name>/predictions/<pred_name>/:
    plot.png      curva con bandas de incertidumbre
    data.csv      x, y_base, y_conservative, y_std
"""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np

from dataset_processor import build_feature_vector, parse_unit


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def parse_aggregate(agg_str):
    tokens = agg_str.split("_")
    return [parse_unit(t) for t in tokens[:3]]


def predict_with_uncertainty(bundle, feat_vec: np.ndarray):
    """
    Devuelve:
        y_base  (P,)  predicción media sin corrección
        y_cons  (P,)  predicción conservadora
        y_std   (P,)  incertidumbre propagada al espacio de curvas
    """
    gprs       = bundle["gprs"]
    pca        = bundle["pca"]
    scaler     = bundle["scaler"]
    correction = bundle.get("correction", np.zeros(pca.n_components_))
    n_comp     = bundle["n_components"]
    components = pca.components_   # (n_comp, P)

    feat_sc    = scaler.transform(feat_vec.reshape(1, -1))

    comps_mean = np.zeros((1, n_comp))
    comps_std  = np.zeros((1, n_comp))
    for i, gpr in enumerate(gprs):
        mu, sigma        = gpr.predict(feat_sc, return_std=True)
        comps_mean[0, i] = mu[0]
        comps_std[0, i]  = sigma[0]

    y_base = pca.inverse_transform(comps_mean)[0]          # (P,)
    y_cons = y_base - correction                            # (P,)

    # Propagar incertidumbre: std_curve ≈ sqrt( sum_i (std_i * component_i)^2 )
    var_curve = np.einsum("ni,ip->np", comps_std**2, components**2)
    y_std     = np.sqrt(var_curve)[0]                      # (P,)

    return y_base, y_cons, y_std


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main(args):
    exp_dir    = Path("./experiments") / args.exp_name
    model_path = exp_dir / "gpr_model.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en {model_path}\n"
            f"Entrena primero con: python gpr_model.py --exp_name {args.exp_name}")

    bundle = joblib.load(model_path)
    x_axis = bundle["x_axis"]

    # ── Métricas del experimento ──
    exp_info = {}
    results_path = exp_dir / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            exp_info = json.load(f)

    # ── Feature vector ──
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

    # ── Predicción ──
    y_base, y_cons, y_std = predict_with_uncertainty(bundle, feat_vec)

    # ── Nombre canónico de la predicción ──
    pred_name = (f"pred_{args.floors}F"
                 f"-{args.position}"
                 f"-{args.diaphragm}"
                 f"-{args.direction}"
                 f"-ax{args.asym_x}"
                 f"-ay{args.asym_y}"
                 f"-{args.agg.replace('_', '.')}")

    # ── Carpeta de salida ──
    out_dir = exp_dir / "predictions" / pred_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Consola ──
    print(f"\nExperimento : {args.exp_name}")
    print(f"Predicción  : {pred_name}")
    if exp_info:
        mb = exp_info.get("metrics_no_bias", {})
        mc = exp_info.get("metrics_conserv", {})
        print(f"LOO R²      : {mb.get('r2', '?'):.4f}  "
              f"(conserv: {mc.get('r2', '?'):.4f})")
        print(f"LOO below%  : {mb.get('pct_below', '?'):.1f}%  "
              f"(conserv: {mc.get('pct_below', '?'):.1f}%)")

    print(f"\nCurva base      → min={y_base.min():.4f}  max={y_base.max():.4f}")
    print(f"Curva conserv.  → min={y_cons.min():.4f}  max={y_cons.max():.4f}")
    print(f"Incertidumbre   → std media={y_std.mean():.4f}  max={y_std.max():.4f}")
    print(f"Corrección media aplicada: {(y_base - y_cons).mean():.4f}")

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(12, 5))

    # Bandas de incertidumbre sobre la media
    ax.fill_between(x_axis,
                    y_base - 2*y_std, y_base + 2*y_std,
                    alpha=0.12, color="steelblue", label="±2σ")
    ax.fill_between(x_axis,
                    y_base - y_std,   y_base + y_std,
                    alpha=0.25, color="steelblue", label="±1σ")

    # Margen entre base y conservadora
    ax.fill_between(x_axis, y_cons, y_base,
                    alpha=0.15, color="orange", label="Margen conservador")

    ax.plot(x_axis, y_base, lw=2, color="steelblue",
            ls="--", label="GPR media")
    ax.plot(x_axis, y_cons, lw=2, color="green",
            label="GPR conservadora")

    title = (f"{args.floors}F · {args.position} · Diaph:{args.diaphragm} · "
             f"Dir:{args.direction} · AsymX:{args.asym_x} · AsymY:{args.asym_y}\n"
             f"Agregado: {args.agg}  |  exp: {args.exp_name}")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out_png = out_dir / "plot.png"
    plt.savefig(out_png, dpi=120)
    if args.plot:
        plt.show()
    plt.close()
    print(f"\nPlot guardado en {out_png}")

    # ── CSV ──
    out_csv = out_dir / "data.csv"
    with open(out_csv, "w") as f:
        f.write("x,y_base,y_conservative,y_std\n")
        for xi, yb, yc, ys in zip(x_axis, y_base, y_cons, y_std):
            f.write(f"{xi:.6f},{yb:.6f},{yc:.6f},{ys:.6f}\n")
    print(f"CSV guardado en {out_csv}")


# ─────────────────────────────────────────────
# ARGS
# ─────────────────────────────────────────────

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
    parser.add_argument("--agg",       default="ISOL_ISOL_ISOL",
                        help="Descriptor del agregado, e.g. '3R11_2R11_1R11'")
    parser.add_argument("--plot",      action="store_true",
                        help="Mostrar el plot en pantalla además de guardarlo")
    args = parser.parse_args()
    main(args)