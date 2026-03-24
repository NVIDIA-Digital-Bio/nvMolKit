from typing import Any, List
from rdkit.Chem import Mol

def UFFOptimizeMoleculesConfs(
    molecules: List[Mol],
    maxIters: int = 1000,
    vdwThresholds: Any = ...,
    ignoreInterfragInteractions: Any = ...,
    hardwareOptions: Any = ...,
) -> List[List[float]]: ...
