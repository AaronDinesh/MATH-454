import numpy as np
import matplotlib.pyplot as plt


timings = np.array([[20.9396, 21.0079, 21.0466, 20.9563, 20.9144],
                    [10.6571, 10.6404, 10.6398, 10.6721, 10.64],
                    [5.51616, 5.53141, 5.52483, 5.76215, 5.51526],
                    [5.41882, 5.43919, 5.41834, 5.43967, 5.44626],
                    [5.44962, 5.46128, 5.46647, 5.45319, 5.46141],
                    [5.65488, 5.67623, 5.63882, 5.64934, 5.6647],
                    [5.65453, 5.69836, 5.66414, 5.66973, 5.67682],
                    [5.28774, 5.62511, 5.61849, 5.60934, 5.60663],
                    [5.87675, 5.89786, 5.89188, 5.87737, 5.86821],
                    [6.29961, 6.29828, 6.28863, 6.27223, 6.26965],
                    [6.59748, 6.6028, 6.61373, 6.60426, 6.58785]])




mean_timings = np.mean(timings, axis=1)
std_timings = np.std(timings, axis=1)
std_errors = std_timings / np.sqrt(timings.shape[1])

# CPU timing
cpu_timing = 2.704444e+01

# X-axis: thread counts
threads = 2 ** np.arange(11) #Get 0-10

# Find index of minimum mean timing
min_idx = np.argmin(mean_timings)

# Plot
plt.figure(figsize=(6, 4))
plt.errorbar(threads, mean_timings, yerr=std_errors, fmt='-o', capsize=4, label='Mean timing')


# Faint dots for individual runs
for i, thread in enumerate(threads):
    plt.scatter([thread]*timings.shape[1], timings[i], color='gray', alpha=0.3)

# Highlight the minimum mean timing
plt.plot(threads[min_idx], mean_timings[min_idx], 'o', color='red', markersize=8, label=f'Minimum timing ({mean_timings[min_idx]:.2f} s)', zorder=100)

# Horizontal CPU line
plt.axhline(cpu_timing, color='black', linestyle='--', label=f'CPU time ({cpu_timing:.2f} s)')

plt.xscale('log', base=2)
plt.xticks(threads, threads)
plt.xlabel('Number of Threads')
plt.ylabel('Timing')
plt.title('Timing vs. Number of Threads')
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.legend()
plt.tight_layout()

# Save for LaTeX
plt.savefig("timing_vs_threads.pdf", format='pdf', bbox_inches='tight')
plt.close()