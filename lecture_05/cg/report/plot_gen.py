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
    timing_vals = np.array([87.1554, 76.9707, 74.4244, 93.035, 179.294, 169.863, 222.772])
    speedup = timing_vals[0]/timing_vals
    amdahl = 1 / (f + (1 - f) / p_vals)
    S_inf = 1 / f

    # === PLOT ===
    plt.figure()
    plt.plot(p_vals, amdahl, marker='o', label="Amdahl's Law (Strong Scaling)", markersize=3)
    plt.plot(p_vals, speedup, marker='s', color='orange', label="Measured Speedups", markersize=3)
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

    