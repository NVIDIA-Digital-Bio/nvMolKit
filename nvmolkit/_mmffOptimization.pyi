from typing import Any, List
from rdkit.Chem import Mol

from nvmolkit.types import OptimizerOptions, OptimizerBackend


def MMFFOptimizeMoleculesConfs(
    molecules: List[Mol],
    maxIters: int = 200,
    nonBondedThreshold: float = 100.0,
    hardwareOptions: Any = None,
    optimizerOptions: OptimizerOptions | None = None,
) -> List[List[float]]: ...
