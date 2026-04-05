"""
analyze_experiment.py
=====================
Analiza un experimento GPR entrenado: para cada muestra del dataset
genera la predicción del modelo, la curva conservadora y las bandas
de incertidumbre, y las compara con la curva original.

Uso:
    python analyze_experiment.py --exp_name exp01_baseline
    python analyze_experiment.py --exp_name exp02_conserv --dataset ./processed/dataset_p256.pt

Salida en ./experiments/<exp_name>/analysis/:
    summary.json          métricas por curva (R², RMSE, pct_below)
    overview.png          panel con todas las curvas (original vs pred vs incert.)
    curves/NNN_<name>.png una figura por cada curva del dataset
"""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch


# ─────────────────────────────────────────────
# INFERENCIA CON INCERTIDUMBRE
# ─────────────────────────────────────────────

def predict_with_uncertainty(bundle, X_sc: np.ndarray):
    """
    Dado X_sc (N, n_features) ya escalado, devuelve:
        y_mean  (N, P)  — predicción media
        y_std   (N, P)  — incertidumbre propagada al espacio de curvas
    """
    gprs       = bundle["gprs"]
    pca        = bundle["pca"]
    n_comp     = bundle["n_components"]
    components = pca.components_   # (n_comp, P)

    N = X_sc.shape[0]
    comps_mean = np.zeros((N, n_comp))
    comps_std  = np.zeros((N, n_comp))

    for i, gpr in enumerate(gprs):
        mu, sigma        = gpr.predict(X_sc, return_std=True)
        comps_mean[:, i] = mu
        comps_std[:, i]  = sigma

    # Reconstruir curvas medias
    y_mean = pca.inverse_transform(comps_mean)   # (N, P)

    # Propagar incertidumbre: std_curve ≈ sqrt( sum_i (std_i * component_i)^2 )
    # comps_std: (N, n_comp)  components: (n_comp, P)
    # var_curve[n, p] = sum_i (comps_std[n,i] * components[i,p])^2
    var_curve = np.einsum("ni,ip->np", comps_std**2, components**2)
    y_std     = np.sqrt(var_curve)               # (N, P)

    return y_mean, y_std


# ─────────────────────────────────────────────
# MÉTRICAS POR CURVA
# ─────────────────────────────────────────────

def curve_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_cons: np.ndarray) -> dict:
    """Métricas para una única curva (arrays 1D)."""
    rmse      = float(np.sqrt(((y_true - y_pred)**2).mean()))
    mae       = float(np.abs(y_true - y_pred).mean())
    ss_res    = float(((y_true - y_pred)**2).sum())
    ss_tot    = float(((y_true - y_true.mean())**2).sum())
    r2        = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    pct_below = float((y_cons < y_true).mean() * 100)
    max_over  = float(np.maximum(0, y_cons - y_true).max())
    return {"r2": r2, "rmse": rmse, "mae": mae,
            "pct_below_conserv": pct_below,
            "max_overestim_conserv": max_over}


# ─────────────────────────────────────────────
# PLOT DE UNA CURVA
# ─────────────────────────────────────────────

def plot_curve(ax, x_axis, y_true, y_mean, y_std, y_cons, title, metrics):
    # Bandas de incertidumbre
    ax.fill_between(x_axis,
                    y_mean - 2*y_std, y_mean + 2*y_std,
                    alpha=0.12, color="steelblue", label="±2σ")
    ax.fill_between(x_axis,
                    y_mean - y_std,   y_mean + y_std,
                    alpha=0.25, color="steelblue", label="±1σ")

    # Zona conservadora vs real
    ax.fill_between(x_axis, y_cons, y_true,
                    where=(y_cons < y_true),
                    alpha=0.10, color="green")
    ax.fill_between(x_axis, y_cons, y_true,
                    where=(y_cons >= y_true),
                    alpha=0.15, color="red")

    ax.plot(x_axis, y_true, lw=1.8, color="black",     label="Original")
    ax.plot(x_axis, y_mean, lw=1.5, color="steelblue",
            ls="--",                                     label="GPR media")
    ax.plot(x_axis, y_cons, lw=1.5, color="green",
            ls=":",                                      label="Conservadora")

    ax.set_title(title, fontsize=7)
    ax.grid(alpha=0.3)

    # Métricas en esquina
    info = (f"R²={metrics['r2']:.3f}  RMSE={metrics['rmse']:.3f}\n"
            f"below={metrics['pct_below_conserv']:.1f}%  "
            f"maxOver={metrics['max_overestim_conserv']:.3f}")
    ax.text(0.02, 0.03, info, transform=ax.transAxes,
            fontsize=6, va="bottom", color="dimgray",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main(args):
    exp_dir  = Path("./experiments") / args.exp_name
    model_path = exp_dir / "gpr_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo en {model_path}")

    # ── Carpeta de análisis ──
    out_dir = exp_dir / "analysis"
    out_dir.mkdir(exist_ok=True)
    curves_dir = out_dir / "curves"
    curves_dir.mkdir(exist_ok=True)

    print(f"\nExperimento : {args.exp_name}")
    print(f"Análisis    : {out_dir}")

    # ── Cargar modelo ──
    bundle     = joblib.load(model_path)
    scaler     = bundle["scaler"]
    correction = bundle.get("correction", np.zeros(bundle["pca"].n_components_))
    x_axis     = bundle["x_axis"]

    # ── Cargar dataset ──
    # Primero intenta el dataset guardado en el experimento, luego el argumento
    exp_results = exp_dir / "results.json"
    dataset_path = args.dataset
    if dataset_path is None:
        if exp_results.exists():
            with open(exp_results) as f:
                meta = json.load(f)
            dataset_path = meta.get("dataset", None)
        if dataset_path is None:
            raise ValueError(
                "No se pudo determinar el dataset. "
                "Pasa --dataset ./processed/dataset_pXXX.pt")

    print(f"Dataset     : {dataset_path}")
    ds       = torch.load(dataset_path, weights_only=False)
    X        = ds["X"].numpy()
    Y        = ds["Y"].numpy()
    filenames = ds.get("filenames", [f"curva_{i}" for i in range(len(X))])

    # ── Predicción de todas las curvas ──
    X_sc            = scaler.transform(X)
    y_mean, y_std   = predict_with_uncertainty(bundle, X_sc)
    y_cons          = y_mean - correction   # (N, P)

    print(f"Predicciones: {len(X)} curvas")

    # ── Métricas por curva ──
    all_metrics = []
    for i in range(len(X)):
        m = curve_metrics(Y[i], y_mean[i], y_cons[i])
        m["index"]    = i
        m["filename"] = filenames[i]
        all_metrics.append(m)

    # Métricas globales
    r2_vals   = [m["r2"]   for m in all_metrics if not np.isnan(m["r2"])]
    rmse_vals = [m["rmse"] for m in all_metrics]
    below_vals= [m["pct_below_conserv"] for m in all_metrics]

    global_metrics = {
        "n_curves":       len(X),
        "r2_mean":        float(np.mean(r2_vals)),
        "r2_min":         float(np.min(r2_vals)),
        "rmse_mean":      float(np.mean(rmse_vals)),
        "rmse_max":       float(np.max(rmse_vals)),
        "pct_below_mean": float(np.mean(below_vals)),
        "pct_below_min":  float(np.min(below_vals)),
        "curves":         [{k: round(v, 6) if isinstance(v, float) else v
                            for k, v in m.items()}
                           for m in all_metrics],
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(global_metrics, f, indent=2)

    print(f"\nMétricas globales (sobre todo el dataset):")
    print(f"  R² medio         : {global_metrics['r2_mean']:.4f}  "
          f"(min: {global_metrics['r2_min']:.4f})")
    print(f"  RMSE medio       : {global_metrics['rmse_mean']:.4f}  "
          f"(max: {global_metrics['rmse_max']:.4f})")
    print(f"  below% medio     : {global_metrics['pct_below_mean']:.1f}%  "
          f"(min: {global_metrics['pct_below_min']:.1f}%)")

    # ── Plot individual por curva ──
    print(f"\nGenerando plots individuales...")
    for i, m in enumerate(all_metrics):
        fname  = Path(filenames[i]).stem[:50]
        title  = f"{i:03d} · {fname}"
        fig, ax = plt.subplots(figsize=(10, 4))
        plot_curve(ax, x_axis, Y[i], y_mean[i], y_std[i], y_cons[i], title, m)
        ax.set_xlabel("X"); ax.set_ylabel("Y")
        handles, labels = ax.get_legend_handles_labels()
        # Eliminar duplicados del fill_between
        seen, h2, l2 = set(), [], []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen.add(l); h2.append(h); l2.append(l)
        ax.legend(h2, l2, fontsize=7, loc="upper right")
        plt.tight_layout()
        safe_name = f"{i:03d}_" + "".join(c if c.isalnum() or c in "-_" else "_" for c in fname)
        plt.savefig(curves_dir / f"{safe_name}.png", dpi=100)
        plt.close()
        print(f"  {i+1}/{len(X)}", end="\r")
    print()

    # ── Overview: panel con todas las curvas ──
    print("Generando overview...")
    N    = len(X)
    ncols = 6
    nrows = int(np.ceil(N / ncols))
    fig  = plt.figure(figsize=(ncols * 3.5, nrows * 2.8))
    gs   = gridspec.GridSpec(nrows, ncols, figure=fig,
                             hspace=0.55, wspace=0.35)

    for i in range(N):
        ax = fig.add_subplot(gs[i // ncols, i % ncols])
        plot_curve(ax, x_axis, Y[i], y_mean[i], y_std[i], y_cons[i],
                   f"{i:03d} · {Path(filenames[i]).stem[:25]}", all_metrics[i])
        ax.tick_params(labelsize=5)
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            seen, h2, l2 = set(), [], []
            for h, l in zip(handles, labels):
                if l not in seen:
                    seen.add(l); h2.append(h); l2.append(l)

    # Leyenda global arriba
    fig.legend(h2, l2, loc="upper center", ncol=len(l2),
               fontsize=8, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(
        f"{args.exp_name}  —  R²={global_metrics['r2_mean']:.3f}  "
        f"RMSE={global_metrics['rmse_mean']:.4f}  "
        f"below%={global_metrics['pct_below_mean']:.1f}%",
        fontsize=10, y=1.04)

    plt.savefig(out_dir / "overview.png", dpi=100, bbox_inches="tight")
    plt.close()

    print(f"\n✓ Análisis completo guardado en {out_dir}/")
    print(f"  overview.png          — panel con las {N} curvas")
    print(f"  curves/               — {N} plots individuales")
    print(f"  summary.json          — métricas por curva")


# ─────────────────────────────────────────────
# ARGS
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", required=True,
                        help="Nombre del experimento a analizar")
    parser.add_argument("--dataset",  default=None,
                        help="Ruta al dataset.pt (opcional si el experimento "
                             "ya tiene la ruta guardada en results.json)")
    args = parser.parse_args()
    main(args)