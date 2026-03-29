import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import re

def normalize_algorithm_name(name):
    name = name.strip()
    if name.lower() == 'row_pipeline':
        return 'Dimension Order Pipeline'
    return name

def normalize_operation_name(op):
    op_l = op.strip().lower()
    if op_l in ('bcast', 'broadcast'):
        return 'Broadcast'
    if op_l == 'reduce':
        return 'Reduce'
    return op

def infer_section_name_from_filename(filename):
    base = os.path.basename(filename)
    match = re.search(r'(2D|3D)[-_ ]*(Bcast|Broadcast|Reduce)', base, flags=re.IGNORECASE)
    if not match:
        return 'Unknown'
    dim = match.group(1).upper()
    op = normalize_operation_name(match.group(2))
    return f"{dim} {op}"

def parse_section_lines(lines, section_name):
    data = {section_name: {}}
    current_algo = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.endswith(':') and line[0].isalpha():
            algo_name = normalize_algorithm_name(line[:-1])
            current_algo = algo_name
            if current_algo not in data[section_name]:
                data[section_name][current_algo] = {'processors': [], 'times': []}
            continue

        if ':' in line and current_algo:
            try:
                proc_str, times_str = line.split(':', 1)
                proc_str = proc_str.strip()
                if not proc_str.isdigit():
                    continue
                processors = int(proc_str)
                times = [float(t) for t in times_str.strip().split()]
                data[section_name][current_algo]['processors'].append(processors)
                data[section_name][current_algo]['times'].append(times)
            except ValueError:
                continue

    return data

def parse_combined_lines(lines):
    data = {}
    current_section = None
    current_algo = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        section_match = re.match(r'^(2D|3D)\s+(Bcast|Broadcast|Reduce)\s*:?$', line, flags=re.IGNORECASE)
        if section_match:
            dim = section_match.group(1).upper()
            op = normalize_operation_name(section_match.group(2))
            current_section = f"{dim} {op}"
            data.setdefault(current_section, {})
            current_algo = None
            continue

        if line.endswith(':') and line[0].isalpha() and current_section:
            algo_name = normalize_algorithm_name(line[:-1])
            current_algo = algo_name
            if current_algo not in data[current_section]:
                data[current_section][current_algo] = {'processors': [], 'times': []}
            continue

        if ':' in line and current_section and current_algo:
            try:
                proc_str, times_str = line.split(':', 1)
                proc_str = proc_str.strip()
                if not proc_str.isdigit():
                    continue
                processors = int(proc_str)
                times = [float(t) for t in times_str.strip().split()]
                data[current_section][current_algo]['processors'].append(processors)
                data[current_section][current_algo]['times'].append(times)
            except ValueError:
                continue

    return data

def merge_data(base_data, new_data):
    for section, algos in new_data.items():
        base_data.setdefault(section, {})
        for algo, values in algos.items():
            if algo not in base_data[section]:
                base_data[section][algo] = {'processors': [], 'times': []}
            base_data[section][algo]['processors'].extend(values['processors'])
            base_data[section][algo]['times'].extend(values['times'])

def parse_times_file(filename='times.txt'):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            lines = f.readlines()

        has_embedded_sections = any(
            re.search(r'\b(2D|3D)\b\s+(Bcast|Broadcast|Reduce)\b', ln, flags=re.IGNORECASE)
            for ln in lines
        )

        if has_embedded_sections:
            return parse_combined_lines(lines)

        inferred_section = infer_section_name_from_filename(filename)
        return parse_section_lines(lines, inferred_section)

    if filename != 'times.txt':
        print(f"Error: {filename} not found.")
        return {}

    # Fallback for the repository's split timing files.
    split_files = {
        '2D Broadcast': ['2D-Broadcast-times.txt'],
        '2D Reduce': ['2D-Reduce-times.txt'],
        '3D Broadcast': ['3D-Broadcast-times.txt'],
        '3D Reduce': ['3D-Reduce-times.txt', '3D-reduce-times.txt'],
    }

    merged = {}
    for section_name, candidates in split_files.items():
        source_file = next((f for f in candidates if os.path.exists(f)), None)
        if source_file is None:
            continue

        with open(source_file, 'r') as f:
            section_lines = f.readlines()

        section_data = parse_section_lines(section_lines, section_name)
        merge_data(merged, section_data)

    if not merged:
        print("Error: no timing files found (expected times.txt or split *-times.txt files).")

    return merged

def calculate_avg_times(data):
    processed = {}
    for section, algos in data.items():
        processed[section] = {}
        for algo, values in algos.items():
            if not values['processors']: continue
            avg_times = [np.mean(t) for t in values['times']]
            processed[section][algo] = {
                'processors': values['processors'],
                'avg_times': avg_times
            }
    return processed

def plot_performance(data, output_dir='plots'):
    Path(output_dir).mkdir(exist_ok=True)
    colors = {'Brute': '#e74c3c', 'Recursive': '#3498db', 'BFS': '#2ecc71',
              'Dimension Order Pipeline': '#f39c12', 'Row_pipeline': '#f39c12',
              'SQRT': '#9b59b6', 'Naive': '#95a5a6'}
    markers = {'Brute': 'o', 'Recursive': 's', 'BFS': '^',
               'Dimension Order Pipeline': 'D', 'Row_pipeline': 'D',
               'SQRT': 'v', 'Naive': 'x'}

    for section, algos in data.items():
        if not algos: continue
        
        # 1. Combined Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        for algo, values in algos.items():
            ax.plot(values['processors'], values['avg_times'], 
                   marker=markers.get(algo, 'o'), color=colors.get(algo, 'black'),
                   label=algo, linewidth=2)
            
        ax.set_xlabel('Number of Processors')
        ax.set_ylabel('Execution Time (seconds)')
        ax.set_title(f'{section} - Performance Comparison')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        filename = f"{section.replace(' ', '_').lower()}_combined.png"
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{filename}", dpi=300)
        plt.close()
        print(f"Generated {filename}")
        
        # 2. No Brute Plot
        if 'Brute' in algos and len(algos) > 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            for algo, values in algos.items():
                if algo == 'Brute': continue
                ax.plot(values['processors'], values['avg_times'], 
                       marker=markers.get(algo, 'o'), color=colors.get(algo, 'black'),
                       label=algo, linewidth=2)
            
            ax.set_xlabel('Number of Processors')
            ax.set_ylabel('Execution Time (seconds)')
            ax.set_title(f'{section} - Performance (No Brute)')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            
            filename = f"{section.replace(' ', '_').lower()}_no_brute.png"
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{filename}", dpi=300)
            plt.close()
            print(f"Generated {filename}")

if __name__ == "__main__":
    raw_data = parse_times_file()
    if raw_data:
        avg_data = calculate_avg_times(raw_data)
        plot_performance(avg_data)
