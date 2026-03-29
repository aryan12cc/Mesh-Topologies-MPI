import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

output_dir = Path("chunking_plots")
output_dir.mkdir(exist_ok=True)

PROCESSORS = 64
MESSAGE_SIZE = 2**22
MESSAGE_SIZE_MB = MESSAGE_SIZE / (1024 * 1024)

CHUNK_SIZES_BYTES = [
    1024,
    4096,
    16384,
    65536,
    262144,
    1048576,
    2097152,
    4194304
]

CHUNK_LABELS = ['1KB', '4KB', '16KB', '64KB', '256KB', '1MB', '2MB', '4MB']

def parse_chunking_data(filename='chunking_times.txt'):
    """Parse the chunking times data file"""
    data_2d = {'naive': [], 'chunked': {}}
    data_3d = {'naive': [], 'chunked': {}}

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith('#') or line.startswith('=') or line.startswith('-') or line.startswith('Chunk') or line.startswith('Fixed') or line.startswith('Format') or line.startswith('Message'):
                continue

            parts = line.split()
            if len(parts) == 6:
                dimension = parts[0]
                method = parts[1]
                chunk_label = parts[2]
                times = [float(parts[3]), float(parts[4]), float(parts[5])]

                chunk_size_bytes = CHUNK_SIZES_BYTES[CHUNK_LABELS.index(chunk_label)]

                if dimension == '2D':
                    if method == 'naive':
                        data_2d['naive'].extend(times)
                    else:
                        data_2d['chunked'][chunk_size_bytes] = times
                elif dimension == '3D':
                    if method == 'naive':
                        data_3d['naive'].extend(times)
                    else:
                        data_3d['chunked'][chunk_size_bytes] = times

    return data_2d, data_3d

data_2d, data_3d = parse_chunking_data()

def plot_chunk_size_comparison(dimension='2D'):
    """Plot execution time vs chunk size"""
    data = data_2d if dimension == '2D' else data_3d

    fig, ax = plt.subplots(figsize=(14, 8))

    naive_avg = np.mean(data['naive'])
    ax.axhline(y=naive_avg, color='red', linestyle='--', linewidth=2.5,
               label=f'Naive (no chunking): {naive_avg:.4f}s', alpha=0.8)

    x_indices = range(len(CHUNK_SIZES_BYTES))
    chunked_avgs = [np.mean(data['chunked'][cs]) for cs in CHUNK_SIZES_BYTES]
    chunked_stds = [np.std(data['chunked'][cs]) for cs in CHUNK_SIZES_BYTES]

    ax.plot(x_indices, chunked_avgs, marker='o', linewidth=3, markersize=10,
            color='#2ecc71', label='Chunked Broadcast')
    ax.errorbar(x_indices, chunked_avgs, yerr=chunked_stds, fmt='none',
                ecolor='#2ecc71', capsize=5, alpha=0.5)

    optimal_idx = np.argmin(chunked_avgs)
    optimal_chunk = CHUNK_LABELS[optimal_idx]
    optimal_time = chunked_avgs[optimal_idx]
    speedup = naive_avg / optimal_time

    ax.plot(optimal_idx, optimal_time, marker='*', markersize=20,
            color='gold', markeredgecolor='black', markeredgewidth=2,
            label=f'Optimal: {optimal_chunk} ({speedup:.2f}x speedup)')

    ax.set_xticks(x_indices)
    ax.set_xticklabels(CHUNK_LABELS, rotation=45, ha='right')
    ax.set_xlabel('Chunk Size', fontsize=13, fontweight='bold')
    ax.set_ylabel('Execution Time (seconds)', fontsize=13, fontweight='bold')
    ax.set_title(f'{dimension} Broadcast Performance vs Chunk Size\n' +
                 f'Fixed: {PROCESSORS} processors, Message size: {MESSAGE_SIZE_MB:.0f}MB',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    filename = f"{dimension.lower()}_chunk_variation.png"
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=300)
    plt.close()
    print(f"Generated {filename}")

def plot_speedup_analysis(dimension='2D'):
    """Plot speedup factor vs chunk size"""
    data = data_2d if dimension == '2D' else data_3d

    fig, ax = plt.subplots(figsize=(14, 8))

    naive_avg = np.mean(data['naive'])
    x_indices = range(len(CHUNK_SIZES_BYTES))
    speedups = [naive_avg / np.mean(data['chunked'][cs]) for cs in CHUNK_SIZES_BYTES]

    colors = ['#e74c3c' if s < 1.5 else '#f39c12' if s < 2.0 else '#27ae60' if s < 2.5 else '#2ecc71'
              for s in speedups]
    bars = ax.bar(x_indices, speedups, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    for i, (bar, speedup) in enumerate(zip(bars, speedups)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{speedup:.2f}x',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='No speedup')

    optimal_idx = np.argmax(speedups)
    optimal_speedup = speedups[optimal_idx]
    bars[optimal_idx].set_edgecolor('gold')
    bars[optimal_idx].set_linewidth(4)

    ax.set_xticks(x_indices)
    ax.set_xticklabels(CHUNK_LABELS, rotation=45, ha='right')
    ax.set_xlabel('Chunk Size', fontsize=13, fontweight='bold')
    ax.set_ylabel('Speedup Factor (Naive / Chunked)', fontsize=13, fontweight='bold')
    ax.set_title(f'{dimension} Chunking Speedup Analysis\n' +
                 f'Fixed: {PROCESSORS} processors, Message size: {MESSAGE_SIZE_MB:.0f}MB',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_dir / f'{dimension.lower()}_speedup_vs_chunk.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {dimension.lower()}_speedup_vs_chunk.png")

def plot_combined_2d_3d():
    """Plot 2D and 3D comparison on same graph"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    x_indices = range(len(CHUNK_SIZES_BYTES))

    for data, dimension, color in [(data_2d, '2D', '#3498db'), (data_3d, '3D', '#e67e22')]:
        naive_avg = np.mean(data['naive'])
        ax1.axhline(y=naive_avg, color=color, linestyle='--', linewidth=2,
                   label=f'{dimension} Naive', alpha=0.6)

        chunked_avgs = [np.mean(data['chunked'][cs]) for cs in CHUNK_SIZES_BYTES]
        ax1.plot(x_indices, chunked_avgs, marker='o', linewidth=2.5, markersize=8,
                color=color, label=f'{dimension} Chunked')

    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(CHUNK_LABELS, rotation=45, ha='right')
    ax1.set_xlabel('Chunk Size', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title('Execution Time Comparison', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    for data, dimension, color in [(data_2d, '2D', '#3498db'), (data_3d, '3D', '#e67e22')]:
        naive_avg = np.mean(data['naive'])
        speedups = [naive_avg / np.mean(data['chunked'][cs]) for cs in CHUNK_SIZES_BYTES]
        ax2.plot(x_indices, speedups, marker='s', linewidth=2.5, markersize=8,
                color=color, label=f'{dimension} Grid')

    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='No speedup')
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(CHUNK_LABELS, rotation=45, ha='right')
    ax2.set_xlabel('Chunk Size', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Speedup Factor', fontsize=12, fontweight='bold')
    ax2.set_title('Speedup Comparison', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    fig.suptitle(f'2D vs 3D Chunking Performance ({PROCESSORS} processors, {MESSAGE_SIZE_MB:.0f}MB message)',
                 fontsize=15, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(output_dir / 'combined_2d_3d_chunk_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: combined_2d_3d_chunk_analysis.png")

def create_summary_table():
    """Create detailed summary table"""
    lines = ["="*80, "CHUNK SIZE VARIATION ANALYSIS", "="*80]
    lines.append(f"\nFixed Parameters:")
    lines.append(f"  Number of Processors: {PROCESSORS}")
    lines.append(f"  Message Size: {MESSAGE_SIZE_MB:.2f} MB ({MESSAGE_SIZE} bytes)")
    lines.append(f"  Grid Configuration: 8x8 (2D) or 4x4x4 (3D)")
    lines.append("\n" + "="*80)

    for dimension, data in [('2D', data_2d), ('3D', data_3d)]:
        lines.append(f"\n{dimension} Results:")
        lines.append("-" * 80)

        naive_avg = np.mean(data['naive'])
        lines.append(f"\nNaive Broadcast Time: {naive_avg:.6f} seconds")
        lines.append(f"\nChunk Size Analysis:")
        lines.append(f"{'Chunk Size':<12} {'Avg Time (s)':<15} {'Speedup':<10} {'Performance'}")
        lines.append("-" * 80)

        rows = []
        for chunk_label, chunk_size in zip(CHUNK_LABELS, CHUNK_SIZES_BYTES):
            chunked_avg = np.mean(data['chunked'][chunk_size])
            speedup = naive_avg / chunked_avg

            if speedup < 1.5:
                perf = "Poor"
            elif speedup < 2.0:
                perf = "Fair"
            elif speedup < 2.5:
                perf = "Good"
            else:
                perf = "Excellent"

            rows.append((chunk_label, chunked_avg, speedup, perf))

        best_chunk, _, best_speedup, _ = max(rows, key=lambda row: row[2])

        for chunk_label, chunked_avg, speedup, perf in rows:
            marker = " ★" if chunk_label == best_chunk else ""
            speedup_str = f"{speedup:.2f}x"
            lines.append(f"{chunk_label:<12} {chunked_avg:<15.6f} {speedup_str:<10} {perf}{marker}")

        lines.append(f"\nOptimal Configuration: {best_chunk} chunks ({best_speedup:.2f}x speedup)")

    lines.append("\n" + "="*80)

    report = "\n".join(lines)
    (output_dir / 'chunk_variation_summary.txt').write_text(report)
    print(f"✓ Saved: chunk_variation_summary.txt")
    print("\n" + report)

if __name__ == "__main__":
    print("\n" + "="*80)
    print("Generating Chunk Size Variation Analysis")
    print(f"Fixed: {PROCESSORS} processors, {MESSAGE_SIZE_MB:.0f}MB message")
    print("="*80 + "\n")

    print("Generating plots...")
    plot_chunk_size_comparison('2D')
    plot_chunk_size_comparison('3D')
    plot_speedup_analysis('2D')
    plot_speedup_analysis('3D')
    plot_combined_2d_3d()

    print("\nGenerating summary...")
    create_summary_table()

    print("\n" + "="*80)
    print("Analysis complete!")
    print(f"Total plots: 5")
    print(f"Location: {output_dir}/ directory")
    print("="*80 + "\n")
