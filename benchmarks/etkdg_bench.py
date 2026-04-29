# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ETKDG conformer generation benchmark comparing nvmolkit GPU embedding against RDKit.

Drives :func:`nvmolkit.embedMolecules.EmbedMolecules` and (optionally) RDKit's
``rdDistGeom.EmbedMultipleConfs`` over the same input set, reports per-method
wall-clock timings and (when validation is enabled) MMFF94 energy deltas
between the two implementations.

Usage:
    python etkdg_bench.py --smiles data/chembl_10k.smi --num_mols 200 --confs_per_mol 10
    python etkdg_bench.py --sdf data/MPCONF196.sdf --confs_per_mol 5 --no_rdkit
    python etkdg_bench.py --pickle prepared_mols.pkl --num_mols 1000 --batch_size 512 --batches_per_gpu 4
"""

import argparse
import gc
import random
import statistics
import sys
from typing import Callable

import nvtx
from bench_utils import (
    clone_mols_with_conformers,
    load_pickle,
    load_sdf,
    load_smiles,
    prep_mols,
    time_it as _time_it,
)
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom


def time_it(func: Callable, runs: int = 1, warmups: int = 0, gpu_sync: bool = False) -> tuple[float, float]:
    """Time ``func`` and return ``(mean_ms, std_ms)``."""
    result = _time_it(func, runs=runs, warmups=warmups, gpu_sync=gpu_sync)
    return result.mean_ms, result.std_ms


def _conformer_count(mols: list[Chem.Mol]) -> int:
    return sum(m.GetNumConformers() for m in mols)


def _mmff_energies(mol: Chem.Mol) -> list[float]:
    """Return MMFF94 energies for each conformer in ``mol``; failures contribute 0.0."""
    energies: list[float] = []
    try:
        props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")
    except Exception:
        return [0.0] * mol.GetNumConformers()
    if props is None:
        return [0.0] * mol.GetNumConformers()
    for conf in mol.GetConformers():
        try:
            ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf.GetId())
            energies.append(float(ff.CalcEnergy()) if ff is not None else 0.0)
        except Exception:
            energies.append(0.0)
    return energies


def _energy_diff_summary(
    rdkit_mols: list[Chem.Mol],
    nvmolkit_mols: list[Chem.Mol],
) -> tuple[float, float, int]:
    """Mean / median energy difference (RDKit - nvmolkit) and the number of paired conformers.

    Conformers where either side has zero energy (failed/missing) are skipped.
    """
    deltas: list[float] = []
    for rd_mol, nv_mol in zip(rdkit_mols, nvmolkit_mols):
        rd_energies = _mmff_energies(rd_mol)
        nv_energies = _mmff_energies(nv_mol)
        paired = min(len(rd_energies), len(nv_energies))
        for i in range(paired):
            if rd_energies[i] == 0.0 or nv_energies[i] == 0.0:
                continue
            deltas.append(rd_energies[i] - nv_energies[i])
    if not deltas:
        return float("nan"), float("nan"), 0
    return statistics.mean(deltas), statistics.median(deltas), len(deltas)


@nvtx.annotate("bench_nvmolkit_etkdg", color="red")
def bench_nvmolkit(
    mols: list[Chem.Mol],
    params,
    confs_per_mol: int,
    max_iters: int,
    hardware_options,
    runs: int,
    warmup: bool,
) -> tuple[float, float, list[Chem.Mol]]:
    """Benchmark nvmolkit ``EmbedMolecules``; return ``(mean_ms, std_ms, last_run_mols)``."""
    from nvmolkit.embedMolecules import EmbedMolecules

    last_run_mols: list[list[Chem.Mol]] = [[]]

    @nvtx.annotate("etkdg_nvmolkit_run", color="orange")
    def run() -> None:
        cloned = clone_mols_with_conformers(mols)
        EmbedMolecules(cloned, params, confs_per_mol, max_iters, hardware_options)
        last_run_mols[0] = cloned

    if warmup:
        warmup_mols = clone_mols_with_conformers(mols[: min(4, len(mols))])
        with nvtx.annotate("etkdg_nvmolkit_warmup", color="purple"):
            EmbedMolecules(warmup_mols, params, 1, max_iters, hardware_options)

    avg_ms, std_ms = time_it(run, runs=runs, warmups=0, gpu_sync=True)
    return avg_ms, std_ms, last_run_mols[0]


@nvtx.annotate("bench_rdkit_etkdg", color="green")
def bench_rdkit(
    mols: list[Chem.Mol],
    params,
    confs_per_mol: int,
    runs: int,
    warmup: bool,
) -> tuple[float, float, list[Chem.Mol]]:
    """Benchmark RDKit ``EmbedMultipleConfs``; return ``(mean_ms, std_ms, last_run_mols)``."""
    last_run_mols: list[list[Chem.Mol]] = [[]]

    @nvtx.annotate("etkdg_rdkit_run", color="yellow")
    def run() -> None:
        cloned = clone_mols_with_conformers(mols)
        for mol in cloned:
            rdDistGeom.EmbedMultipleConfs(mol, numConfs=confs_per_mol, params=params)
        last_run_mols[0] = cloned

    if warmup:
        warmup_mol = Chem.RWMol(mols[0])
        rdDistGeom.EmbedMultipleConfs(warmup_mol, numConfs=1, params=params)

    avg_ms, std_ms = time_it(run, runs=runs, warmups=0, gpu_sync=False)
    return avg_ms, std_ms, last_run_mols[0]


def _build_etkdg_params(max_iterations: int, num_threads: int, seed: int) -> rdDistGeom.EmbedParameters:
    params = rdDistGeom.ETKDGv3()
    params.useRandomCoords = True
    if max_iterations > 0:
        params.maxIterations = max_iterations
    params.numThreads = num_threads
    params.randomSeed = seed
    return params


def _build_hardware_options(
    batch_size: int,
    batches_per_gpu: int,
    prep_threads: int,
    num_gpus: int,
):
    from nvmolkit.types import HardwareOptions

    return HardwareOptions(
        preprocessingThreads=prep_threads,
        batchSize=batch_size,
        batchesPerGpu=batches_per_gpu,
        gpuIds=list(range(num_gpus)) if num_gpus > 0 else None,
    )


CSV_HEADER = (
    "method,input_file,input_type,num_mols,confs_per_mol,max_iterations,"
    "batch_size,batches_per_gpu,prep_threads,num_gpus,nvmolkit_config_source,"
    "rdkit_threads,time_ms,std_ms,conformers_generated,mean_energy_diff,median_energy_diff,"
    "energy_diff_pairs"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ETKDG conformer generation benchmark: nvmolkit vs RDKit",
    )
    parser.add_argument("--smiles", "-s", help="Path to SMILES file with molecules")
    parser.add_argument("--sdf", help="Path to SDF file (alternative to --smiles)")
    parser.add_argument("--pickle", help="Path to pickled RDKit binary molecules (alternative to --smiles)")
    parser.add_argument("--num_mols", "-n", type=int, default=0, help="Max number of molecules (default: 0 = all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling and ETKDG (default: 42)")
    parser.add_argument(
        "--sanitize", action="store_true", dest="sanitize", help="Sanitize molecules during parsing (default)"
    )
    parser.add_argument("--no_sanitize", action="store_false", dest="sanitize", help="Skip sanitization at parse time")
    parser.set_defaults(sanitize=True)

    parser.add_argument("--confs_per_mol", "-c", type=int, default=10, help="Conformers per molecule (default: 10)")
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=-1,
        help="Maximum ETKDG iterations; -1 = automatic (default: -1)",
    )

    parser.add_argument("--runs", "-r", type=int, default=1, help="Number of timing runs (default: 1)")
    parser.add_argument("--warmup", action="store_true", dest="warmup", help="Perform a warmup run (default)")
    parser.add_argument("--no_warmup", action="store_false", dest="warmup", help="Skip warmup")
    parser.set_defaults(warmup=True)

    parser.add_argument("--no_nvmolkit", action="store_true", help="Skip nvmolkit benchmark")
    parser.add_argument("--no_rdkit", action="store_true", help="Skip RDKit benchmark")
    parser.add_argument(
        "--rdkit_threads",
        type=int,
        default=1,
        help="Threads passed to RDKit ETKDG via params.numThreads (default: 1)",
    )

    parser.add_argument("--batch_size", "-b", type=int, default=1024, help="nvmolkit batch size (default: 1024)")
    parser.add_argument(
        "--batches_per_gpu", type=int, default=-1, help="nvmolkit concurrent batches per GPU (-1 = library default)"
    )
    parser.add_argument(
        "--prep_threads", type=int, default=-1, help="nvmolkit preprocessing threads (-1 = library default)"
    )
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use (default: 1)")

    parser.add_argument(
        "--validate", action="store_true", dest="validate", help="Compute MMFF energy diffs vs RDKit (default)"
    )
    parser.add_argument(
        "--no_validate", action="store_false", dest="validate", help="Skip energy validation"
    )
    parser.set_defaults(validate=True)

    parser.add_argument("--output", "-o", default=None, help="Optional path to write the CSV results")

    args = parser.parse_args()

    input_paths = [p for p in (args.smiles, args.sdf, args.pickle) if p]
    if not input_paths:
        print("Error: One of --smiles, --sdf, or --pickle is required")
        sys.exit(1)
    if len(input_paths) > 1:
        print("Error: --smiles, --sdf, and --pickle are mutually exclusive")
        sys.exit(1)
    if args.num_gpus <= 0:
        print("Error: --num_gpus must be >= 1")
        sys.exit(1)
    if args.no_nvmolkit and args.no_rdkit:
        print("Error: cannot disable both nvmolkit and RDKit")
        sys.exit(1)
    input_file = input_paths[0]
    if args.smiles:
        input_type = "smiles"
    elif args.sdf:
        input_type = "sdf"
    else:
        input_type = "pickle"

    print("\nConfiguration:")
    print(f"  Input: {input_file} ({input_type})")
    print(f"  Max molecules: {args.num_mols if args.num_mols > 0 else 'all'}")
    print(f"  Conformers per mol: {args.confs_per_mol}")
    print(f"  Max iterations: {args.max_iterations if args.max_iterations > 0 else 'auto'}")
    print(f"  Runs: {args.runs}")
    print(f"  Warmup: {args.warmup}")
    print(f"  Validate (MMFF energies): {args.validate}")
    print(f"  Run nvmolkit: {not args.no_nvmolkit}")
    print(f"  Run RDKit: {not args.no_rdkit}")
    if not args.no_rdkit:
        print(f"  RDKit threads: {args.rdkit_threads}")
    if not args.no_nvmolkit:
        print(f"  nvmolkit hardware:")
        print(f"    batch_size: {args.batch_size}")
        print(f"    batches_per_gpu: {args.batches_per_gpu if args.batches_per_gpu > 0 else 'auto'}")
        print(f"    prep_threads: {args.prep_threads if args.prep_threads > 0 else 'auto'}")
        print(f"    num_gpus: {args.num_gpus}")

    print("\nLoading molecules...")
    if args.smiles:
        raw_mols = load_smiles(args.smiles, args.num_mols, args.sanitize, seed=args.seed)
    elif args.sdf:
        raw_mols = load_sdf(args.sdf, args.num_mols, seed=args.seed, sanitize=args.sanitize)
    else:
        raw_mols = load_pickle(args.pickle, args.num_mols, seed=args.seed)
    if not raw_mols:
        print("Error: No valid molecules loaded")
        sys.exit(1)

    print("\nPreparing molecules (AddHs / sanitize / clear conformers)...")
    mols = prep_mols(raw_mols)
    if not mols:
        print("Error: No molecules survived preparation")
        sys.exit(1)
    print(f"  {len(mols)} molecules ready")

    params = _build_etkdg_params(args.max_iterations, args.rdkit_threads, args.seed)

    results: dict[str, tuple[float, float, list[Chem.Mol]]] = {}

    torch_module = None
    config_source = "cli"
    if not args.no_nvmolkit:
        try:
            import torch

            from nvmolkit.types import HardwareOptions

            torch_module = torch
            gpu_ids = list(range(args.num_gpus))

            hardware_options = _build_hardware_options(
                args.batch_size, args.batches_per_gpu, args.prep_threads, args.num_gpus
            )

            torch.cuda.cudart().cudaProfilerStart()
            print("\nRunning nvmolkit ETKDG benchmark...")
            nv_avg, nv_std, nv_mols = bench_nvmolkit(
                mols,
                params,
                args.confs_per_mol,
                args.max_iterations,
                hardware_options,
                args.runs,
                args.warmup,
            )
            print(f"  nvmolkit:        {nv_avg:10.2f} ms (+/- {nv_std:.2f} ms)")
            results["nvmolkit"] = (nv_avg, nv_std, nv_mols)
            torch.cuda.cudart().cudaProfilerStop()
        except ImportError as exc:
            print(f"  nvmolkit: SKIPPED (import error: {exc})")

    if not args.no_rdkit:
        print("\nRunning RDKit ETKDG benchmark...")
        rd_avg, rd_std, rd_mols = bench_rdkit(mols, params, args.confs_per_mol, args.runs, args.warmup)
        print(f"  RDKit:           {rd_avg:10.2f} ms (+/- {rd_std:.2f} ms)")
        results["rdkit"] = (rd_avg, rd_std, rd_mols)

    if not results:
        print("Error: No benchmarks were run")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("Summary:")
    baseline_ms = results.get("rdkit", (None, None, None))[0]
    for name, (avg_ms, std_ms, _) in results.items():
        speedup = ""
        if baseline_ms is not None and name != "rdkit" and avg_ms > 0:
            speedup = f", {baseline_ms / avg_ms:.1f}x vs RDKit"
        print(f"  {name:20s}: {avg_ms:10.2f} ms (+/- {std_ms:.2f} ms){speedup}")

    energy_mean = float("nan")
    energy_median = float("nan")
    energy_pairs = 0
    if args.validate and "nvmolkit" in results and "rdkit" in results:
        print("\nValidation (MMFF94 energies)...")
        energy_mean, energy_median, energy_pairs = _energy_diff_summary(
            results["rdkit"][2], results["nvmolkit"][2]
        )
        if energy_pairs > 0:
            print(
                f"  RDKit - nvmolkit: mean={energy_mean:.3f}, median={energy_median:.3f} "
                f"kcal/mol over {energy_pairs} paired conformers"
            )
        else:
            print("  No paired conformers with valid energies on both sides")

    if "nvmolkit" in results:
        applied_batch_size = int(hardware_options.batchSize)
        applied_batches_per_gpu = int(hardware_options.batchesPerGpu)
        applied_prep_threads = int(hardware_options.preprocessingThreads)
        applied_num_gpus = len(list(hardware_options.gpuIds)) if hardware_options.gpuIds else args.num_gpus
    else:
        applied_batch_size = args.batch_size
        applied_batches_per_gpu = args.batches_per_gpu
        applied_prep_threads = args.prep_threads
        applied_num_gpus = args.num_gpus

    csv_rows: list[str] = []
    for name, (avg_ms, std_ms, run_mols) in results.items():
        is_nv = name == "nvmolkit"
        batch_size = applied_batch_size if is_nv else "N/A"
        batches_per_gpu = applied_batches_per_gpu if is_nv else "N/A"
        prep_threads = applied_prep_threads if is_nv else "N/A"
        num_gpus = applied_num_gpus if is_nv else "N/A"
        nvmolkit_config_source = config_source if is_nv else "N/A"
        rdkit_threads = args.rdkit_threads if name == "rdkit" else "N/A"
        confs_generated = _conformer_count(run_mols)
        mean_diff = energy_mean if (args.validate and is_nv) else "N/A"
        median_diff = energy_median if (args.validate and is_nv) else "N/A"
        pairs = energy_pairs if (args.validate and is_nv) else "N/A"
        csv_rows.append(
            f"{name},{input_file},{input_type},{len(mols)},{args.confs_per_mol},"
            f"{args.max_iterations},{batch_size},{batches_per_gpu},{prep_threads},{num_gpus},"
            f"{nvmolkit_config_source},{rdkit_threads},{avg_ms:.2f},{std_ms:.2f},"
            f"{confs_generated},{mean_diff},{median_diff},{pairs}"
        )

    print("\n\nCSV Results:")
    print(CSV_HEADER)
    for row in csv_rows:
        print(row)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(CSV_HEADER + "\n")
            for row in csv_rows:
                fh.write(row + "\n")
        print(f"\nWrote results to {args.output}")

    if torch_module is not None:
        torch_module.cuda.synchronize()
        torch_module.cuda.empty_cache()
        torch_module.cuda.ipc_collect()
    gc.collect()


if __name__ == "__main__":
    main()
