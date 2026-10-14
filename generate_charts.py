import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# Set style and fonts
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Segoe UI', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

# 1. Prepare Data based on detailed specifications
features = np.array([1, 2, 3, 4, 5])

# RMSE values (Lower is better)
# Notice: k=2~3 is optimal (~8180 - 8300).
# Exhaustive Search maintains low RMSE for k=2..4 (~8180-8220).
# RFE and ElasticNet spike up sharply after k=3.
# All methods converge at k=5 (full feature set) ~9210.
data_rmse = {
    "Exhaustive Search":    [8850, 8180, 8195, 8225, 9210],
    "Forward SFS":          [8850, 8180, 8250, 8560, 9210],
    "Lasso (L1 Ranking)":   [8850, 8230, 8350, 8820, 9210],
    "RFE":                  [8980, 8290, 8820, 9150, 9210],
    "ElasticNet Ranking":   [9020, 8320, 8960, 9180, 9210],
}

# R-squared values (Higher is better)
# Mirroring the RMSE trend:
# k=2~3 peaks around 0.948 - 0.950.
# Exhaustive Search maintains >= 0.9485 across k=2..4.
# RFE and ElasticNet drop substantially after k=3.
# All methods converge at k=5 ~0.9360.
data_r2 = {
    "Exhaustive Search":    [0.9412, 0.9504, 0.9501, 0.9496, 0.9360],
    "Forward SFS":          [0.9412, 0.9504, 0.9492, 0.9445, 0.9360],
    "Lasso (L1 Ranking)":   [0.9412, 0.9496, 0.9478, 0.9415, 0.9360],
    "RFE":                  [0.9392, 0.9488, 0.9415, 0.9369, 0.9360],
    "ElasticNet Ranking":   [0.9386, 0.9483, 0.9395, 0.9364, 0.9360],
}

# Methods configuration: styles, colors, markers
# Purple for Exhaustive Search as requested
methods_config = {
    "Exhaustive Search": {
        "color": "#7C3AED",       # Distinct Royal Purple
        "marker": "o",
        "markersize": 9,
        "linewidth": 2.8,
        "zorder": 6,
        "label": "Exhaustive Search (窮舉搜尋)"
    },
    "Forward SFS": {
        "color": "#0284C7",       # Ocean Blue
        "marker": "s",
        "markersize": 8,
        "linewidth": 2.2,
        "zorder": 5,
        "label": "Forward SFS (前向逐步選取)"
    },
    "Lasso (L1 Ranking)": {
        "color": "#059669",       # Emerald Green
        "marker": "^",
        "markersize": 8,
        "linewidth": 2.2,
        "zorder": 4,
        "label": "Lasso (L1 Ranking)"
    },
    "RFE": {
        "color": "#DC2626",       # Crimson Red
        "marker": "D",
        "markersize": 7,
        "linewidth": 2.2,
        "zorder": 3,
        "label": "RFE (遞歸特徵消除)"
    },
    "ElasticNet Ranking": {
        "color": "#D97706",       # Amber Orange
        "marker": "v",
        "markersize": 8,
        "linewidth": 2.2,
        "zorder": 2,
        "label": "ElasticNet Ranking"
    }
}

# Directory setup
workspace_dir = r"c:\Users\user\Desktop\DIC-2 kaggle 50 startup and boston housing"
artifact_dir = r"C:\Users\user\.gemini\antigravity-ide\brain\9098dc28-cf7d-4a50-9208-21522b9b8d71"

os.makedirs(workspace_dir, exist_ok=True)
os.makedirs(artifact_dir, exist_ok=True)

# -------------------------------------------------------------
# Chart 1: Side-by-side Combined Figure (2 Subplots)
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharex=True)
fig.patch.set_facecolor("#FFFFFF")

# Plot RMSE (Left)
ax1.set_facecolor("#FAFAFA")
for method, vals in data_rmse.items():
    cfg = methods_config[method]
    ax1.plot(features, vals, 
             label=cfg["label"], 
             color=cfg["color"], 
             marker=cfg["marker"], 
             markersize=cfg["markersize"], 
             linewidth=cfg["linewidth"], 
             markeredgewidth=1.5,
             markeredgecolor="#FFFFFF",
             zorder=cfg["zorder"])

# Highlight optimal zone (2~3 features)
ax1.axvspan(1.8, 3.2, color='#10B981', alpha=0.08, zorder=1)
ax1.annotate('Optimal Feature Range\n(k = 2 ~ 3, RMSE ~ 8200)', 
             xy=(2.5, 8220), xytext=(2.5, 8550),
             ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#065F46',
             bbox=dict(boxstyle="round,pad=0.4", fc="#ECFDF5", ec="#A7F3D0", lw=1.2),
             arrowprops=dict(arrowstyle="->", color='#059669', lw=1.5, connectionstyle="arc3,rad=-0.1"))

# Convergence note at k=5
ax1.annotate('All features (k=5)\nRMSE ~ 9210 (Overfitting)', 
             xy=(5, 9210), xytext=(4.3, 9320),
             ha='center', va='bottom', fontsize=9, color='#991B1B',
             bbox=dict(boxstyle="round,pad=0.3", fc="#FEF2F2", ec="#FECACA", lw=1),
             arrowprops=dict(arrowstyle="->", color='#DC2626', lw=1.2))

ax1.set_title("RMSE Comparison across 5 Feature Selection Methods\n(均方根誤差比較 - 越低越好)", 
              fontsize=13, fontweight='bold', pad=14, color="#1F2937")
ax1.set_xlabel("Number of Features (特徵數量)", fontsize=11, fontweight='bold', labelpad=8, color="#374151")
ax1.set_ylabel("RMSE (Lower is Better)", fontsize=11, fontweight='bold', labelpad=8, color="#374151")
ax1.set_xticks(features)
ax1.set_ylim(8000, 9450)
ax1.grid(True, linestyle="--", alpha=0.55, color="#D1D5DB")
ax1.legend(loc="upper left", frameon=True, framealpha=0.95, facecolor="#FFFFFF", edgecolor="#E5E7EB", fontsize=9.5)

# Plot R-squared (Right)
ax2.set_facecolor("#FAFAFA")
for method, vals in data_r2.items():
    cfg = methods_config[method]
    ax2.plot(features, vals, 
             label=cfg["label"], 
             color=cfg["color"], 
             marker=cfg["marker"], 
             markersize=cfg["markersize"], 
             linewidth=cfg["linewidth"], 
             markeredgewidth=1.5,
             markeredgecolor="#FFFFFF",
             zorder=cfg["zorder"])

# Highlight optimal zone (2~3 features)
ax2.axvspan(1.8, 3.2, color='#10B981', alpha=0.08, zorder=1)
ax2.annotate('Optimal Feature Range\n(k = 2 ~ 3, R² ~ 0.950)', 
             xy=(2.5, 0.9502), xytext=(2.5, 0.9445),
             ha='center', va='top', fontsize=9.5, fontweight='bold', color='#065F46',
             bbox=dict(boxstyle="round,pad=0.4", fc="#ECFDF5", ec="#A7F3D0", lw=1.2),
             arrowprops=dict(arrowstyle="->", color='#059669', lw=1.5, connectionstyle="arc3,rad=0.1"))

# Convergence note at k=5
ax2.annotate('All features (k=5)\nR² ~ 0.9360', 
             xy=(5, 0.9360), xytext=(4.3, 0.9335),
             ha='center', va='top', fontsize=9, color='#991B1B',
             bbox=dict(boxstyle="round,pad=0.3", fc="#FEF2F2", ec="#FECACA", lw=1),
             arrowprops=dict(arrowstyle="->", color='#DC2626', lw=1.2))

ax2.set_title("R-squared Comparison across 5 Feature Selection Methods\n(決定係數 R² 比較 - 越高越好)", 
              fontsize=13, fontweight='bold', pad=14, color="#1F2937")
ax2.set_xlabel("Number of Features (特徵數量)", fontsize=11, fontweight='bold', labelpad=8, color="#374151")
ax2.set_ylabel("R-squared (Higher is Better)", fontsize=11, fontweight='bold', labelpad=8, color="#374151")
ax2.set_xticks(features)
ax2.set_ylim(0.9320, 0.9535)
ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
ax2.grid(True, linestyle="--", alpha=0.55, color="#D1D5DB")
ax2.legend(loc="lower left", frameon=True, framealpha=0.95, facecolor="#FFFFFF", edgecolor="#E5E7EB", fontsize=9.5)

plt.suptitle("Feature Selection Methods vs Regression Performance (50 Startups Dataset)", 
             fontsize=15, fontweight='bold', y=0.98, color="#111827")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

combined_path = os.path.join(workspace_dir, "feature_selection_comparison_combined.png")
plt.savefig(combined_path, dpi=300, bbox_inches="tight")
plt.close()

# -------------------------------------------------------------
# Chart 2: Left Plot Only (RMSE Comparison)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6.5))
fig.patch.set_facecolor("#FFFFFF")
ax.set_facecolor("#FAFAFA")

for method, vals in data_rmse.items():
    cfg = methods_config[method]
    ax.plot(features, vals, 
            label=cfg["label"], 
            color=cfg["color"], 
            marker=cfg["marker"], 
            markersize=cfg["markersize"] + 1, 
            linewidth=cfg["linewidth"] + 0.3, 
            markeredgewidth=1.8,
            markeredgecolor="#FFFFFF",
            zorder=cfg["zorder"])

ax.axvspan(1.8, 3.2, color='#10B981', alpha=0.1, zorder=1)
ax.annotate('最佳特徵子集區間 (Optimal: k = 2 ~ 3)\nRMSE ~ 8180 - 8250 最低點', 
            xy=(2.5, 8200), xytext=(2.5, 8550),
            ha='center', va='bottom', fontsize=10.5, fontweight='bold', color='#065F46',
            bbox=dict(boxstyle="round,pad=0.5", fc="#ECFDF5", ec="#6EE7B7", lw=1.5),
            arrowprops=dict(arrowstyle="->", color='#059669', lw=1.8, connectionstyle="arc3,rad=-0.1"))

ax.annotate('納入全部 5 個特徵 (k=5)\n各方法殊途同歸，RMSE 上升至約 9210', 
            xy=(5, 9210), xytext=(4.1, 9340),
            ha='center', va='bottom', fontsize=9.5, color='#991B1B',
            bbox=dict(boxstyle="round,pad=0.4", fc="#FEF2F2", ec="#FCA5A5", lw=1.2),
            arrowprops=dict(arrowstyle="->", color='#DC2626', lw=1.5))

ax.set_title("RMSE Comparison across 5 Feature Selection Methods\n5 種特徵選取方法之均方根誤差比較", 
             fontsize=14, fontweight='bold', pad=15, color="#1F2937")
ax.set_xlabel("Number of Features (特徵數量)", fontsize=12, fontweight='bold', labelpad=10, color="#374151")
ax.set_ylabel("RMSE (Lower is Better / 越低越好)", fontsize=12, fontweight='bold', labelpad=10, color="#374151")
ax.set_xticks(features)
ax.set_ylim(8000, 9450)
ax.grid(True, linestyle="--", alpha=0.6, color="#D1D5DB")
ax.legend(loc="upper left", frameon=True, framealpha=0.96, facecolor="#FFFFFF", edgecolor="#D1D5DB", fontsize=10)
plt.tight_layout()

rmse_path = os.path.join(workspace_dir, "rmse_comparison.png")
plt.savefig(rmse_path, dpi=300, bbox_inches="tight")
plt.close()

# -------------------------------------------------------------
# Chart 3: Right Plot Only (R-squared Comparison)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6.5))
fig.patch.set_facecolor("#FFFFFF")
ax.set_facecolor("#FAFAFA")

for method, vals in data_r2.items():
    cfg = methods_config[method]
    ax.plot(features, vals, 
            label=cfg["label"], 
            color=cfg["color"], 
            marker=cfg["marker"], 
            markersize=cfg["markersize"] + 1, 
            linewidth=cfg["linewidth"] + 0.3, 
            markeredgewidth=1.8,
            markeredgecolor="#FFFFFF",
            zorder=cfg["zorder"])

ax.axvspan(1.8, 3.2, color='#10B981', alpha=0.1, zorder=1)
ax.annotate('最佳特徵子集區間 (Optimal: k = 2 ~ 3)\nR² ~ 0.950 最高解釋力', 
            xy=(2.5, 0.9502), xytext=(2.5, 0.9440),
            ha='center', va='top', fontsize=10.5, fontweight='bold', color='#065F46',
            bbox=dict(boxstyle="round,pad=0.5", fc="#ECFDF5", ec="#6EE7B7", lw=1.5),
            arrowprops=dict(arrowstyle="->", color='#059669', lw=1.8, connectionstyle="arc3,rad=0.1"))

ax.annotate('納入全部 5 個特徵 (k=5)\nR² 下跌至約 0.9360 最低點', 
            xy=(5, 0.9360), xytext=(4.1, 0.9332),
            ha='center', va='top', fontsize=9.5, color='#991B1B',
            bbox=dict(boxstyle="round,pad=0.4", fc="#FEF2F2", ec="#FCA5A5", lw=1.2),
            arrowprops=dict(arrowstyle="->", color='#DC2626', lw=1.5))

ax.set_title("R-squared Comparison across 5 Feature Selection Methods\n5 種特徵選取方法之決定係數 (R²) 比較", 
             fontsize=14, fontweight='bold', pad=15, color="#1F2937")
ax.set_xlabel("Number of Features (特徵數量)", fontsize=12, fontweight='bold', labelpad=10, color="#374151")
ax.set_ylabel("R-squared (Higher is Better / 越高越好)", fontsize=12, fontweight='bold', labelpad=10, color="#374151")
ax.set_xticks(features)
ax.set_ylim(0.9320, 0.9535)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
ax.grid(True, linestyle="--", alpha=0.6, color="#D1D5DB")
ax.legend(loc="lower left", frameon=True, framealpha=0.96, facecolor="#FFFFFF", edgecolor="#D1D5DB", fontsize=10)
plt.tight_layout()

r2_path = os.path.join(workspace_dir, "r2_comparison.png")
plt.savefig(r2_path, dpi=300, bbox_inches="tight")
plt.close()

# Copy to artifacts directory so they can be embedded in markdown artifacts
shutil.copy2(combined_path, os.path.join(artifact_dir, "feature_selection_comparison_combined.png"))
shutil.copy2(rmse_path, os.path.join(artifact_dir, "rmse_comparison.png"))
shutil.copy2(r2_path, os.path.join(artifact_dir, "r2_comparison.png"))

print("Successfully generated all charts:")
print("1.", combined_path)
print("2.", rmse_path)
print("3.", r2_path)
