from nvmolkit.fingerprints import MorganFingerprintGenerator as nvmolMorganGen
from nvmolkit.similarity import crossTanimotoSimilarity
from rdkit.Chem import MolFromSmiles
import torch
from rdkit.ML.Cluster.Butina import ClusterData
from nvmolkit.clustering import butina as butina_nvmol
import pandas as pd
import sys

def get_distance_matrix(mols):
    nvmol_gen = nvmolMorganGen(radius=2, fpSize=1024)
    nvmol_fps = nvmol_gen.GetFingerprints(mols, 10)
    sim_matrix = crossTanimotoSimilarity(nvmol_fps).torch()
    return 1.0 - sim_matrix

def resize_and_fill(distance_mat: torch.Tensor, want_size):
    current_size = distance_mat.shape[0]
    if current_size >= want_size:
        return distance_mat[:want_size, :want_size].contiguous()
    full_mat = torch.rand(want_size, want_size, dtype=distance_mat.dtype, device=distance_mat.device)
    full_mat = torch.abs(full_mat - full_mat.T).clip(0.01, 0.99)
    full_mat.fill_diagonal_(0.0)
    full_mat[:current_size, :current_size] = distance_mat

def timeIt(func, runs=3, warmups=1):
    import time
    for _ in range(warmups):
        res = func()
    times = []
    for _ in range(runs):
        start = time.time_ns()
        res = func()
        end = time.time_ns()
        times.append(end - start)
    avg_time = sum(times) / runs
    std_time = (sum((t - avg_time) ** 2 for t in times) / runs) ** 0.5
    return avg_time / 1.0e6, std_time / 1.0e6  # return in milliseconds
def bench_rdkit(data, threshold):
    return timeIt(lambda: ClusterData(data, len(data), threshold, isDistData=True, reordering=True))

def bench_nvmolkit(data, threshold):
    return timeIt(lambda: butina_nvmol(data, threshold))

if __name__ == "__main__":
    input_data = "/data/chembl_size_splits/chembl_40-60.smi"
    with open(input_data, "r") as f:
        smis = [line.strip() for line in f.readlines()]
    mols = [MolFromSmiles(smi, sanitize=True) for smi in smis[:21000]]
    mols = [mol for mol in mols if mol is not None]

    dists = get_distance_matrix(mols)


    sizes = [5000, 10000, 15000, 20000]
    # cutoffs = [10e-8, 0.1, 0.2, 1.1]
    cutoffs = [0.2]
    results = []


    for cutoff in cutoffs:
        for size in sizes:
            print(f"Running size {size} cutoff {cutoff}")
            dist_mat = resize_and_fill(dists, size)
            dist_mat_numpy = dist_mat.cpu().numpy()

            rdkit_time, rdk_std = bench_rdkit(dist_mat_numpy, cutoff)
            nvmol_time, nvmol_std = timeIt(lambda: butina_nvmol(dist_mat, cutoff))

            results.append({
                "size": size,
                "cutoff": cutoff,
                "rdkit_time_ms": rdkit_time,
                "rdkit_std_ms": rdk_std,
                "nvmol_time_ms": nvmol_time,
                "nvmol_std_ms": nvmol_std
            })

    df = pd.DataFrame(results)
    df.to_csv('results.csv', index=False)
    import math
    import matplotlib.pyplot as plt
    plt.figure()
    for cutoff in df['cutoff'].unique():
        subset = df[df['cutoff'] == cutoff]
        plt.errorbar(subset['size'], subset['rdkit_time_ms'], yerr=subset['rdkit_std_ms'] * 1.95 / math.sqrt(3), capsize=10, label=f'RDKit Cutoff {cutoff}', fmt='-o')
        plt.errorbar(subset['size'], subset['nvmol_time_ms'], yerr=subset['nvmol_std_ms']* 1.95 / math.sqrt(3), capsize=10, label=f'nvMolKit Cutoff {cutoff}', fmt='-o')
    plt.xlabel('Number of Molecules')
    plt.ylabel('Time (ms)')
    plt.title('Butina Clustering Performance Comparison')
    plt.legend()
    plt.show()

    # Speedup plot
    plt.figure()
    for cutoff in [0.1, 0.2]:
        #for cutoff in df['cutoff'].unique():
        subset = df[df['cutoff'] == cutoff]
        speedup = subset['rdkit_time_ms'] / subset['nvmol_time_ms']
        plt.plot(subset['size'], speedup, '-o', label=f'Cutoff {cutoff}')
    plt.xlabel('Number of Molecules')
    plt.ylabel('Speedup (RDKit / nvMolKit)')
    plt.title('Butina Clustering Speedup')
    plt.yscale('log')
    plt.legend()
    plt.show()

    # Ratio of input size to time
    plt.figure()
    for cutoff in [0.1]:
        subset = df[df['cutoff'] == cutoff]
        rdkit_ratio = subset['size'] ** 2 / subset['rdkit_time_ms']
        nvmol_ratio = subset['size'] ** 2 / subset['nvmol_time_ms']
        plt.plot(subset['size'], rdkit_ratio, '-o', label=f'RDKit Cutoff {cutoff}')
        plt.plot(subset['size'], nvmol_ratio, '-o', label=f'nvMolKit Cutoff {cutoff}')
    plt.legend()
    plt.show()
