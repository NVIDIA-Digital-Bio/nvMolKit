from typing import Any, List
from rdkit.Chem import Mol

def MMFFOptimizeMoleculesConfs(
    molecules: List[Mol],
    maxIters: int = 200,
    nonBondedThreshold: float = 100.0,
    hardwareOptions: Any = None
) -> List[List[float]]: ...
