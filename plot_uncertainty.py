# plot_uncertainty.py
import joblib
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset_processor import build_feature_vector, parse_unit

def parse_aggregate(agg_str):
    tokens = agg_str.split("_")
    return [parse_unit(t) for t in tokens[:3]]

# ── Cargar modelo ──
bundle = joblib.load("./experiments/exp01_baseline/gpr_model.pkl")
gprs, pca, scaler = bundle["gprs"], bundle["pca"], bundle["scaler"]
x_axis     = bundle["x_axis"]
correction = bundle.get("correction", np.zeros(len(gprs)))
n_comp     = bundle["n_components"]

# ── Definir caso ──
info = {
    "floors":      2,
    "position":    "C",
    "diaphragm":   "R",
    "direction":   "Y",
    "asym_x":      1,
    "asym_y":      1,
    "agg_units":   parse_aggregate("3R11_2R11_1R11"),
    "is_isolated": False,
}
feat_sc = scaler.transform(build_feature_vector(info).reshape(1, -1))

# ── Predecir media y std por componente PCA ──
comps_mean = np.zeros((1, n_comp))
comps_std  = np.zeros((1, n_comp))

for i, gpr in enumerate(gprs):
    mu, sigma = gpr.predict(feat_sc, return_std=True)
    comps_mean[0, i] = mu[0]
    comps_std[0, i]  = sigma[0]

# ── Propagar incertidumbre al espacio de curvas via PCA ──
# La varianza en el espacio de curvas es V = comps = comps_std² · componentes²
# Aproximación: std_curve ≈ sqrt( sum_i (std_i * pca.components_[i])² )
components = pca.components_   # (n_comp, 256) --> Ojo, 256 es el n_points que he indicad oen dataset_processor.py. Si lo cambias es diferente
var_curve  = (comps_std[0, :, None] * components) ** 2
std_curve  = np.sqrt(var_curve.sum(axis=0))   # (256,)

y_mean = pca.inverse_transform(comps_mean)[0]
y_cons = y_mean - correction

# ── Plot ──
fig, ax = plt.subplots(figsize=(12, 5))

# Bandas de incertidumbre: 1σ, 2σ
ax.fill_between(x_axis,
                y_mean - 2*std_curve, y_mean + 2*std_curve,
                alpha=0.15, color="steelblue", label="±2σ")
ax.fill_between(x_axis,
                y_mean - std_curve, y_mean + std_curve,
                alpha=0.30, color="steelblue", label="±1σ")

ax.plot(x_axis, y_mean, lw=2, color="steelblue", label="Media GPR")
ax.plot(x_axis, y_cons, lw=2, color="green", ls="--", label="Conservadora")

ax.set_xlabel("X"); ax.set_ylabel("Y")
ax.set_title("Predicción GPR con incertidumbre")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("./experiments/exp01_baseline/uncertainty_plot.png", dpi=120)
plt.show()