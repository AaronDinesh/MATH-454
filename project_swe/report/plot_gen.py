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
    f = 0.0739                # Serial fraction
    p_max = 128                # Number of processors
    output_file = "./plots/scaling_laws.pdf"

    # === DATA ===
    p_vals = np.logspace(0, np.log2(p_max), int(np.log2(p_max)) +1, base=2, endpoint=True)
    amdahl = 1 / (f + (1 - f) / p_vals)
    gustafson = p_vals - f * (p_vals - 1)
    S_inf = 1 / f

    # === PLOT ===
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=10)
    plt.loglog(p_vals, amdahl, marker='o', label="Amdahl's Law (Strong Scaling)", markersize=3)
    plt.loglog(p_vals, gustafson, marker='s', color='orange', label="Gustafson's Law (Weak Scaling)", markersize=3)
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
    f = 0.0739                # Serial fraction
    p_max = 128                # Number of processors
    output_file = "./plots/strong_scaling.pdf"

    # === DATA ===
    p_vals = np.logspace(0, np.log2(p_max), int(np.log2(p_max)) + 1, base=2, endpoint=True)
    timing_vals_512 = np.array([0.00290759, 7.2633,  4.32061, 1.84866, 0.990194, 0.604195, 0.389784, 0.537168])
    timing_vals_1024 = np.array([197.479, 85.4909, 29.9112, 15.3219 ,8.26209, 4.59359, 2.59976, 1.89142])
    timing_vals_2048 = np.array([1420.62, 499.618, 267.266, 151.107, 83.0941, 45.6691, 23.5018, 13.4561])

    speedup_512 = timing_vals_512[0]/timing_vals_512
    speedup_1024 = timing_vals_1024[0]/timing_vals_1024
    speedup_2048 = timing_vals_2048[0]/timing_vals_2048    

    strong_efficiency_512 = (timing_vals_512[0] / (timing_vals_512*p_vals))*100
    strong_efficiency_1024 = (timing_vals_1024[0] / (timing_vals_1024*p_vals))*100
    strong_efficiency_2048 = (timing_vals_2048[0] / (timing_vals_2048*p_vals))*100

    amdahl = 1 / (f + (1 - f) / p_vals)
    S_inf = 1 / f
    # === PLOT ===
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)  # equivalent to plt.subplot(111)
    ax.set_xscale('log', base=2)   # base 10 for x-axis
    ax.set_yscale('log', base=10)    # base 2 for y-axis
    plt.plot(p_vals, amdahl, marker='o', label="Amdahl's Law (Strong Scaling)", markersize=3)
    plt.plot(p_vals, speedup_512, marker='s', color='orange', label="512x512", markersize=3)

    plt.plot(p_vals, speedup_1024, marker='s', color='green', label="1024x1024", markersize=3)
    for x, y in zip(p_vals, speedup_1024):
        plt.text(x, y, f"{y:.2f}", ha='center', va='bottom', fontsize=8, rotation=45)

    plt.plot(p_vals, speedup_2048, marker='s', color='blue', label="2048x2048", markersize=3)



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

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)  # equivalent to plt.subplot(111)
    ax.set_xscale('log', base=2)   # base 10 for x-axiss
    plt.plot(p_vals, strong_efficiency_512, marker='s', color='orange', label="512x512", markersize=3)
    for x, y in zip(p_vals, strong_efficiency_512):
        plt.text(x, y, f"{y:.2f}", ha='center', va='bottom', fontsize=8, rotation=45)

    plt.plot(p_vals, strong_efficiency_1024, marker='s', color='green', label="1024x1024", markersize=3)
    for x, y in zip(p_vals, strong_efficiency_1024):
        plt.text(x, y, f"{y:.2f}", ha='center', va='bottom', fontsize=8, rotation=45)

    plt.plot(p_vals, strong_efficiency_2048, marker='s', color='blue', label="2048x2048", markersize=3)
    for x, y in zip(p_vals, strong_efficiency_2048):
        plt.text(x, y, f"{y:.2f}", ha='center', va='bottom', fontsize=8, rotation=45)

    plt.title("Strong Scaling Efficiency")
    plt.xlabel(r"Number of processors ($p$)")
    plt.ylabel(r"Efficiency $E(p)$ %")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("./plots/strong_scaling_efficiency.pdf", bbox_inches='tight')
    print(f"Plot saved as './plots/strong_scaling_efficiency.pdf'")
    plt.show()




strong_scaling()

def weak_scaling():
    # === PARAMETERS ===
    f = 0.0739                # Serial fraction
    p_max = 64                # Number of processors
    output_file = "./plots/weak_scaling.pdf"

    # === DATA ===
    p_vals = np.array([1, 4, 16, 64])
    timing_vals = np.array([0.00287946, 0.0592141, 0.937006, 23.9314])
    gustafson = p_vals - f * (p_vals - 1)

    speedup = (timing_vals[0]*p_vals) / (timing_vals)
    weak_scaling_efficiency = (speedup / p_vals) * 100
    print(f"Weak scaling speedup: {speedup}")
    # === PLOT ===
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)  # equivalent to plt.subplot(111)
    ax.set_xscale('log', base=2)   # base 10 for x-axis
    ax.set_yscale('log', base=10)    # base 2 for y-axis
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

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)  # equivalent to plt.subplot(111)
    ax.set_xscale('log', base=2)
    plt.plot(p_vals, weak_scaling_efficiency, marker='s', color='green', label="Weak Scaling Efficiency", markersize=3)
    for x, y in zip(p_vals, weak_scaling_efficiency):
        plt.text(x, y, f"{y:.4f}", ha='center', va='bottom', fontsize=8, rotation=45)
    plt.title("Weak Scaling Efficiency")
    plt.xlabel(r"Number of processors ($p$)")
    plt.ylabel(r"Efficiency $E(p)$ %")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("./plots/weak_scaling_efficiency.pdf", bbox_inches='tight')
    print(f"Plot saved as './plots/weak_scaling_efficiency.pdf'")
    plt.show()

weak_scaling()