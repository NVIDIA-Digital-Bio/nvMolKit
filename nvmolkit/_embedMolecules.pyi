from typing import List, Optional
from rdkit.Chem import Mol
from rdkit.Chem.rdDistGeom import EmbedParameters

class BatchHardwareOptions:
    """Hardware configuration options for ETKDG embedding."""
    
    numThreads: int
    batchSize: int  
    batchesPerGpu: int
    gpuIds: List[int]
    
    def __init__(self) -> None: ...

def EmbedMolecules(
    molecules: List[Mol],
    params: EmbedParameters,
    confsPerMolecule: int = 1,
    maxIterations: int = -1,
    hardwareOptions: Optional[BatchHardwareOptions] = ...
) -> None: ...
