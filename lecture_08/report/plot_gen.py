import numpy as np
import matplotlib.pyplot as plt

# === SETTINGS FOR LATEX-READY PLOT ===
plt.rcParams.update({
    'text.usetex': True,                 # Use LaTeX for rendering (requires LaTeX installation)
    'font.family': 'serif',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'lines.linewidth': 2,
    'lines.markersize': 4,              # Smaller markers
    'figure.dpi': 300,
    'figure.figsize': (6, 4)            # A4 report friendly
})


def scaling_laws():
    # === PARAMETERS ===
    f = 0.1373                # Serial fraction
    p_max = 64                # Number of processors
    output_file = "./plots/scaling_laws.pdf"

    # === DATA ===
    p_vals = np.logspace(0, np.log2(p_max), int(np.log2(p_max)) +1, base=2, endpoint=True)
    amdahl = 1 / (f + (1 - f) / p_vals)
    gustafson = p_vals - f * (p_vals - 1)
    S_inf = 1 / f

    # === PLOT ===
    plt.figure()
    plt.plot(p_vals, amdahl, marker='o', label="Amdahl's Law (Strong Scaling)", markersize=3)
    plt.plot(p_vals, gustafson, marker='s', color='orange', label="Gustafson's Law (Weak Scaling)", markersize=3)
    plt.axhline(y=S_inf, color='red', linestyle='--', label=rf"Asymptotic limit $1/f \approx {S_inf:.2f}$", linewidth=1.25)
    plt.title("Scaling Predictions")
    plt.xlabel(r"Number of processors ($p$)")
    plt.ylabel(r"Speedup $S(p)$")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Plot saved as '{output_file}'")

    plt.show()

scaling_laws()


def strong_scaling():
    # === PARAMETERS ===
    f = 0.1373                # Serial fraction
    p_max = 64                # Number of processors
    output_file = "./plots/strong_scaling.pdf"

    # === DATA ===
    p_vals = np.logspace(0, np.log2(p_max), int(np.log2(p_max)) + 1, base=2, endpoint=True)
    timing_vals_1000 = np.array([87.1554, 76.9707, 74.4244, 93.035, 179.294, 169.863, 222.772])
    timing_vals_200 = np.array([0.34509, 0.270676, 0.231563, 0.273626, 0.345135, 0.407071, 0.637618])
    speedup_1000 = timing_vals_1000[0]/timing_vals_1000
    speedup_200 = timing_vals_200[0]/timing_vals_200
    amdahl = 1 / (f + (1 - f) / p_vals)
    S_inf = 1 / f
    print(f"Speedup 1000: {speedup_1000}")
    print(f"Speedup 200: {speedup_200}")
    # === PLOT ===
    plt.figure()
    plt.plot(p_vals, amdahl, marker='o', label="Amdahl's Law (Strong Scaling)", markersize=3)
    plt.plot(p_vals, speedup_1000, marker='s', color='orange', label="n1000", markersize=3)
    for x, y in zip(p_vals, speedup_1000):
        plt.text(x, y, f"{y:.2f}", ha='center', va='bottom', fontsize=8, rotation=45)

    plt.plot(p_vals, speedup_200, marker='s', color='green', label="n200", markersize=3)
    for x, y in zip(p_vals, speedup_200):
        plt.text(x, y, f"{y:.2f}", ha='center', va='bottom', fontsize=8, rotation=45)
    plt.axhline(y=S_inf, color='red', linestyle='--', label=rf"Asymptotic limit $1/f \approx {S_inf:.2f}$", linewidth=1.25)
    plt.title("Scaling Predictions vs. Measured Speedups")
    plt.xlabel(r"Number of processors ($p$)")
    plt.ylabel(r"Speedup $S(p)$")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Plot saved as '{output_file}'")
    plt.show()

strong_scaling()

def weak_scaling():
    # === PARAMETERS ===
    f = 0.1373                # Serial fraction
    p_max = 64                # Number of processors
    output_file = "./plots/weak_scaling.pdf"

    # === DATA ===
    p_vals = np.logspace(0, np.log2(p_max), int(np.log2(p_max)) + 1, base=2, endpoint=True)
    timing_vals = np.array([0.0288728, 0.272215, 1.01825, 16.9661, 98.8874, 139.376, 147.508])
    iter_count = np.array([488, 1157, 1731, 3515, 5941, 5941, 5941])
    gustafson = p_vals - f * (p_vals - 1)

    speedup = (timing_vals[0] / iter_count[0]) / (timing_vals / iter_count)
    print(f"Weak scaling speedup: {speedup}")
    # === PLOT ===
    plt.figure()
    plt.plot(p_vals, gustafson, marker='o', label="Gustafson's Law (Weak Scaling)", markersize=3)
    plt.plot(p_vals, speedup, marker='s', color='green', label="Speedup", markersize=3)
    for x, y in zip(p_vals, speedup):
        plt.text(x, y, f"{y:.4f}", ha='center', va='bottom', fontsize=8, rotation=45)
    plt.title("Scaling Predictions vs. Measured Speedups")
    plt.xlabel(r"Number of processors ($p$)")
    plt.ylabel(r"Speedup $S(p)$")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Plot saved as '{output_file}'")
    plt.show()

weak_scaling()