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
    return full_mat

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
    do_rdkit = False
    input_data = "/data/chembl_size_splits/chembl_40-60.smi"
    with open(input_data, "r") as f:
        smis = [line.strip() for line in f.readlines()]
    mols = [MolFromSmiles(smi, sanitize=True) for smi in smis[:40100]]
    mols = [mol for mol in mols if mol is not None]

    dists = get_distance_matrix(mols)


    sizes = [1000, 5000, 10000, 20000, 30000, 40000]
    # cutoffs = [10e-8, 0.1, 0.2, 1.1]
    cutoffs = [1e-10, 0.1, 0.2, 1.1]
    results = []

    try:
        for size in sizes:
            for cutoff in cutoffs:
                if cutoff in (1e-10, 1.1) and size > 20000:
                    continue
                print(f"Running size {size} cutoff {cutoff}")
                dist_mat = resize_and_fill(dists, size)
                if do_rdkit:
                    dist_mat_numpy = dist_mat.cpu().numpy()
                    rdkit_time, rdk_std = bench_rdkit(dist_mat_numpy, cutoff)
                else:
                    rdkit_time = 0.0
                    rdk_std = 0.0
                nvmol_time, nvmol_std = timeIt(lambda: butina_nvmol(dist_mat, cutoff))

                results.append({
                    "size": size,
                    "cutoff": cutoff,
                    "rdkit_time_ms": rdkit_time,
                    "rdkit_std_ms": rdk_std,
                    "nvmol_time_ms": nvmol_time,
                    "nvmol_std_ms": nvmol_std
                })
    except Exception as e:
        print(f"Got exception: {e}, exiting early")
    df = pd.DataFrame(results)
    print(df)
    df.to_csv('results.csv', index=False)

