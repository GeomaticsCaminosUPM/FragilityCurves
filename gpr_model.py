"""
gpr_model.py
============
GPR con PCA para regresión funcional de curvas.
Soporta múltiples experimentos con logging y bias conservador.

Uso:
    python gpr_model.py --exp_name exp01_baseline --n_components 15
    python gpr_model.py --exp_name exp02_conservative --n_components 15 --conservative_bias 0.05
    python gpr_model.py --exp_name exp03_more_pca --n_components 25 --conservative_bias 0.05
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────
# EXPERIMENTO
# ─────────────────────────────────────────────

def run_experiment(args):
    # ── Carpeta del experimento ──
    exp_dir = Path("./experiments") / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'─'*55}")
    print(f"Experimento : {args.exp_name}")
    print(f"Carpeta     : {exp_dir}")
    print(f"{'─'*55}")

    # ── Cargar datos ──
    ds     = torch.load(args.dataset, weights_only=False)
    X      = ds["X"].numpy()
    Y      = ds["Y"].numpy()
    x_axis = ds["x_axis"].numpy()
    print(f"Dataset     : X{X.shape}  Y{Y.shape}")

    # ── PCA ──
    pca   = PCA(n_components=args.n_components)
    Y_pca = pca.fit_transform(Y)
    var_exp = pca.explained_variance_ratio_.cumsum()[-1]
    print(f"PCA {args.n_components} componentes → varianza explicada: {var_exp*100:.2f}%")

    # ── Normalizar X ──
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    # ── Kernel ──
    kernel = Matern(nu=args.matern_nu,
                    length_scale_bounds=(1e-3, 1e3)) + WhiteKernel()

    # ── Entrenar GPRs con todos los datos (modelo final) ──
    print("\nEntrenando modelo final...")
    gprs = []
    for i in range(args.n_components):
        gpr = GaussianProcessRegressor(kernel=kernel,
                                       n_restarts_optimizer=args.n_restarts,
                                       normalize_y=True)
        gpr.fit(X_sc, Y_pca[:, i])
        gprs.append(gpr)
        print(f"  GPR componente {i+1}/{args.n_components} ✓", end="\r")
    print()

    # ── LOO CV ──
    print("LOO cross-validation...")
    loo = LeaveOneOut()
    y_true_all, y_pred_all = [], []

    for k, (train_idx, test_idx) in enumerate(loo.split(X_sc)):
        pca_loo   = PCA(n_components=args.n_components)
        Y_pca_loo = pca_loo.fit_transform(Y[train_idx])
        sc_loo    = StandardScaler()
        X_tr      = sc_loo.fit_transform(X[train_idx])
        X_te      = sc_loo.transform(X[test_idx])

        comps_pred = np.zeros((1, args.n_components))
        for i in range(args.n_components):
            g = GaussianProcessRegressor(kernel=kernel,
                                         n_restarts_optimizer=2,
                                         normalize_y=True)
            g.fit(X_tr, Y_pca_loo[:, i])
            comps_pred[0, i] = g.predict(X_te)[0]

        y_pred = pca_loo.inverse_transform(comps_pred)[0]
        y_true = Y[test_idx][0]
        y_true_all.append(y_true)
        y_pred_all.append(y_pred)
        print(f"  LOO {k+1}/{len(X_sc)}", end="\r")

    print()
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    # ── Bias conservador ──
    # Calcula cuánto está la predicción por encima de la real en promedio
    # y lo resta globalmente para que tienda a quedar por debajo
    overestim = (y_pred_all - y_true_all)   # positivo = predicción por encima
    bias_per_point = overestim.mean(axis=0)  # (256,) bias medio en cada punto

    if args.conservative_bias > 0:
        # Aplicamos el bias observado × factor + margen adicional fijo
        correction = bias_per_point * args.bias_correction_factor + args.conservative_bias
        y_pred_cons = y_pred_all - correction
        print(f"Bias conservador aplicado: corrección media = "
              f"{correction.mean():.4f} (rango [{correction.min():.4f}, {correction.max():.4f}])")
    else:
        y_pred_cons = y_pred_all
        correction  = np.zeros_like(bias_per_point)

    # ── Métricas ──
    def metrics(y_true, y_pred, label=""):
        ss_res = ((y_true - y_pred) ** 2).sum()
        ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum()
        r2   = 1 - ss_res / ss_tot
        rmse = np.sqrt(((y_true - y_pred) ** 2).mean())
        mae  = np.abs(y_true - y_pred).mean()
        # % de puntos donde la predicción está por debajo de la real
        pct_below = (y_pred < y_true).mean() * 100
        if label:
            print(f"\n{label}")
            print(f"  R²        : {r2:.4f}")
            print(f"  RMSE      : {rmse:.6f}")
            print(f"  MAE       : {mae:.6f}")
            print(f"  % below   : {pct_below:.1f}%  (objetivo: >50% para conservador)")
        return {"r2": r2, "rmse": rmse, "mae": mae, "pct_below": pct_below}

    m_base = metrics(y_true_all, y_pred_all,  "── Sin bias (LOO)")
    m_cons = metrics(y_true_all, y_pred_cons, "── Con bias conservador (LOO)")

    # ── Guardar modelo ──
    model_path = exp_dir / "gpr_model.pkl"
    joblib.dump({
        "gprs":         gprs,
        "pca":          pca,
        "scaler":       scaler,
        "x_axis":       x_axis,
        "n_components": args.n_components,
        "correction":   correction,   # se aplica en predict_gpr.py
        "args":         vars(args),
    }, model_path)
    print(f"\nModelo guardado en {model_path}")

    # ── Log JSON ──
    log = {
        "exp_name":          args.exp_name,
        "timestamp":         datetime.now().isoformat(),
        "dataset":           args.dataset,
        "dataset_file":      Path(args.dataset).name,
        "n_components":      args.n_components,
        "var_explained_pct": round(float(var_exp) * 100, 3),
        "matern_nu":         args.matern_nu,
        "n_restarts":        args.n_restarts,
        "conservative_bias": args.conservative_bias,
        "bias_correction_factor": args.bias_correction_factor,
        "metrics_no_bias":   {k: round(float(v), 6) for k, v in m_base.items()},
        "metrics_conserv":   {k: round(float(v), 6) for k, v in m_cons.items()},
    }
    with open(exp_dir / "results.json", "w") as f:
        json.dump(log, f, indent=2)

    # Registro global de todos los experimentos
    global_log = Path("./experiments/all_results.jsonl")
    with open(global_log, "a") as f:
        f.write(json.dumps(log) + "\n")

    print(f"Resultados guardados en {exp_dir / 'results.json'}")

    # ── Plots ──
    _plot_loo(x_axis, y_true_all, y_pred_all, y_pred_cons,
              m_base, m_cons, exp_dir, args)

    return log


# ─────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────

def _plot_loo(x_axis, y_true, y_pred, y_pred_cons,
              m_base, m_cons, exp_dir, args):

    np.random.seed(0)
    idx = np.random.choice(len(y_true), min(6, len(y_true)), replace=False)

    # ── Plot comparativo: sin bias vs con bias ──
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, i in zip(axes.flatten(), idx):
        ax.plot(x_axis, y_true[i],      lw=1.5, color="steelblue",  label="Real")
        ax.plot(x_axis, y_pred[i],      lw=1.5, color="orange",
                ls="--", label=f"GPR (R²={m_base['r2']:.3f})")
        if args.conservative_bias > 0:
            ax.plot(x_axis, y_pred_cons[i], lw=1.5, color="green",
                    ls=":",  label=f"GPR conserv. (↓{args.conservative_bias})")
        ax.fill_between(x_axis, y_pred_cons[i], y_true[i],
                        where=(y_pred_cons[i] < y_true[i]),
                        alpha=0.1, color="green", label="Zona segura")
        ax.fill_between(x_axis, y_pred_cons[i], y_true[i],
                        where=(y_pred_cons[i] >= y_true[i]),
                        alpha=0.1, color="red", label="Sobreestima")
        ax.set_title(f"Muestra {i}", fontsize=9)
        ax.legend(fontsize=6); ax.grid(alpha=0.3)

    title = (f"{args.exp_name} — LOO\n"
             f"Sin bias: R²={m_base['r2']:.3f} RMSE={m_base['rmse']:.4f} "
             f"below={m_base['pct_below']:.1f}%  |  "
             f"Conserv: R²={m_cons['r2']:.3f} RMSE={m_cons['rmse']:.4f} "
             f"below={m_cons['pct_below']:.1f}%")
    plt.suptitle(title, fontsize=9)
    plt.tight_layout()
    plt.savefig(exp_dir / "loo_plot.png", dpi=120)
    plt.close()

    # ── Plot de bias medio por punto ──
    bias = y_pred - y_true
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.plot(x_axis, bias.mean(axis=0),  lw=2, color="orange", label="Bias medio")
    ax.fill_between(x_axis,
                    bias.mean(axis=0) - bias.std(axis=0),
                    bias.mean(axis=0) + bias.std(axis=0),
                    alpha=0.2, color="orange", label="±1 std")
    ax.set_title(f"{args.exp_name} — Bias por punto (pred - real)")
    ax.set_xlabel("X"); ax.set_ylabel("Bias")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(exp_dir / "bias_plot.png", dpi=120)
    plt.close()

    print(f"Plots guardados en {exp_dir}/")


# ─────────────────────────────────────────────
# RESUMEN DE EXPERIMENTOS
# ─────────────────────────────────────────────

def print_summary():
    log_path = Path("./experiments/all_results.jsonl")
    if not log_path.exists():
        print("No hay experimentos registrados aún.")
        return

    results = []
    with open(log_path) as f:
        for line in f:
            results.append(json.loads(line))

    W = 120
    print(f"\n{'─'*W}")
    print(f"{'Experimento':<28} {'Dataset':<16} {'n_comp':>6} {'var%':>6} {'bias':>6} "
          f"{'R²':>7} {'RMSE':>8} {'below%':>7} "
          f"{'R²_cons':>8} {'below%_c':>9}")
    print(f"{'─'*W}")
    for r in results:
        mb = r["metrics_no_bias"]
        mc = r["metrics_conserv"]
        # dataset_file puede no existir en logs antiguos
        ds_label = r.get("dataset_file", Path(r["dataset"]).name)
        # Recortar a 15 chars para que quepa en la tabla
        ds_label = ds_label[:15]
        print(f"{r['exp_name']:<28} "
              f"{ds_label:<16} "
              f"{r['n_components']:>6} "
              f"{r['var_explained_pct']:>6.1f} "
              f"{r['conservative_bias']:>6.3f} "
              f"{mb['r2']:>7.4f} "
              f"{mb['rmse']:>8.5f} "
              f"{mb['pct_below']:>7.1f} "
              f"{mc['r2']:>8.4f} "
              f"{mc['pct_below']:>9.1f}")
    print(f"{'─'*W}")


# ─────────────────────────────────────────────
# ARGS
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name",    default=None,
                        help="Nombre del experimento (se usa como carpeta). "
                             "No necesario si solo se usa --summary.")
    parser.add_argument("--dataset",     default="./processed/dataset_p256.pt")
    parser.add_argument("--n_components",type=int,   default=15,
                        help="Componentes PCA")
    parser.add_argument("--matern_nu",   type=float, default=2.5,
                        choices=[0.5, 1.5, 2.5],
                        help="Suavidad del kernel Matérn")
    parser.add_argument("--n_restarts",  type=int,   default=5,
                        help="Reinicios del optimizador del kernel")
    parser.add_argument("--conservative_bias", type=float, default=0.05,
                        help="Margen fijo adicional para predicción conservadora "
                             "(0 = sin bias). Unidades de Y.")
    parser.add_argument("--bias_correction_factor", type=float, default=1.0,
                        help="Factor sobre el bias observado en LOO (1.0 = corrección completa)")
    parser.add_argument("--summary",     action="store_true",
                        help="Mostrar resumen de todos los experimentos y salir")
    args = parser.parse_args()

    if args.summary:
        print_summary()
    elif args.exp_name is None:
        parser.error("--exp_name es obligatorio para entrenar. "
                     "Usa --summary para ver resultados sin entrenar.")
    else:
        run_experiment(args)
        print_summary()