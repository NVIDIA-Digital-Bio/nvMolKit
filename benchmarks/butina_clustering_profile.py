from nvmolkit.fingerprints import MorganFingerprintGenerator as nvmolMorganGen
from nvmolkit.similarity import crossTanimotoSimilarity
from rdkit.Chem import MolFromSmiles
import torch
from nvmolkit.clustering import butina as butina_nvmol
import sys
import nvtx
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

if __name__ == "__main__":
    with nvtx.annotate("Setup", color="blue"):
        size = int(sys.argv[1])
        cutoff = float(sys.argv[2])
        input_data = "/data/chembl_size_splits/chembl_40-60.smi"
        with open(input_data, "r") as f:
            smis = [line.strip() for line in f.readlines()]
        mols = [MolFromSmiles(smi, sanitize=True) for smi in smis[:size+100]]
        mols = [mol for mol in mols if mol is not None][:size]

        dists = get_distance_matrix(mols)
        dist_mat = resize_and_fill(dists, size)
    with nvtx.annotate("Warmup", color="red"):
        warmup = butina_nvmol(dist_mat[:10, :10].contiguous(), 0.2)
    with nvtx.annotate("Clustering", color="green"):
        res = butina_nvmol(dist_mat, cutoff)
