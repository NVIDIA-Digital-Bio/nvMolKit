from typing import Dict, List, Optional, Union
import torch
from rdkit.DataStructs import ExplicitBitVect
from nvmolkit.types import AsyncGpuResult

def CrossTanimotoSimilarityRawBuffers(fp1: Dict, fp2: Dict) -> AsyncGpuResult: ...

def CrossCosineSimilarityRawBuffers(fp1: Dict, fp2: Dict) -> AsyncGpuResult: ...
