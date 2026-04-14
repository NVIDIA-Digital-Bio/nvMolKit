from typing import List

from rdkit.Chem import Mol

from nvmolkit._embedMolecules import BatchHardwareOptions

def UFFOptimizeMoleculesConfs(
    molecules: List[Mol],
    maxIters: int = 1000,
    vdwThresholds: List[float] = ...,
    ignoreInterfragInteractions: List[bool] = ...,
    hardwareOptions: BatchHardwareOptions = ...,
) -> List[List[float]]: ...
