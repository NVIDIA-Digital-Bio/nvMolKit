from typing import List, Optional
from rdkit.Chem import Mol
from rdkit.Chem.rdDistGeom import EmbedParameters

class BatchHardwareOptions: ...

def EmbedMolecules(
    molecules: List[Mol],
    params: EmbedParameters,
    confsPerMolecule: int = 1,
    maxIterations: int = -1,
    hardwareOptions: Optional[BatchHardwareOptions] = ...
) -> None: ...
