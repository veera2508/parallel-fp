import numpy as np
import matplotlib.pyplot as plt

# Use default (light) style
plt.style.use('default')

# Data
dataset_sizes = ['10M', '100M', '1B']
naive_times = [1.3, 14, 140]
mpi_openmp_times = [1, 7, 60]
cuda_times = [0.2, 1, 26]

bar_width = 0.22
x = np.arange(len(dataset_sizes))

colors = ['#FBC02D', '#1976D2', '#43A047']  # Blue, Green, Yellow

fig, ax = plt.subplots(figsize=(10, 6))

bars1 = ax.bar(x - bar_width, naive_times, width=bar_width, color=colors[0], label='Naive', edgecolor='black', linewidth=1)
bars2 = ax.bar(x, mpi_openmp_times, width=bar_width, color=colors[1], label='MPI + OpenMP', edgecolor='black', linewidth=1)
bars3 = ax.bar(x + bar_width, cuda_times, width=bar_width, color=colors[2], label='CUDA', edgecolor='black', linewidth=1)

# Add value labels
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)

# Calculate speedups
cuda_over_cpu = [mpi_openmp_times[i] / cuda_times[i] for i in range(len(dataset_sizes))]
cuda_over_naive = [naive_times[i] / cuda_times[i] for i in range(len(dataset_sizes))]

# Annotate speedups
for i in range(len(dataset_sizes) - 1):
    ax.text(x[i] + bar_width, 20, f'CUDA/CPU: {cuda_over_cpu[i]:.1f}x',
            ha='center', va='bottom', fontsize=10, color='#388E3C', fontweight='bold')
    ax.text(x[i] + bar_width, 30, f'CUDA/Naive: {cuda_over_naive[i]:.1f}x',
            ha='center', va='bottom', fontsize=10, color='black', fontweight='bold')

ax.text(x[i+1] + bar_width, 80, f'CUDA/CPU: {cuda_over_cpu[i+1]:.1f}x',
            ha='center', va='bottom', fontsize=10, color='#388E3C', fontweight='bold')
ax.text(x[i+1] + bar_width, 90, f'CUDA/Naive: {cuda_over_naive[i+1]:.1f}x',
            ha='center', va='bottom', fontsize=10, color='black', fontweight='bold')

# Labels and legend
ax.set_xlabel('Dataset Size', fontsize=13, fontweight='bold')
ax.set_ylabel('Time (seconds)', fontsize=13, fontweight='bold')
ax.set_title('Performance Comparison', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(dataset_sizes, fontsize=12)
ax.legend(fontsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Do NOT use log scale on y axis (default is linear, so nothing to change)

plt.tight_layout()
plt.savefig('performance_comparison.png', dpi=300)
plt.show()