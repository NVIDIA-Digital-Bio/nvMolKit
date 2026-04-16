from typing import Any, List
from rdkit.Chem import Mol

def MMFFOptimizeMoleculesConfs(
    molecules: List[Mol],
    maxIters: int = 200,
    properties: Any = None,
    hardwareOptions: Any = None
) -> List[List[float]]: ...
