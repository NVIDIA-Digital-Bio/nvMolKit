.. SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

NVIDIA nvMolKit Documentation
===================================

nvMolKit Introduction
---------------------

nvMolKit is a CUDA-backed python library for accelerating common RDKit molecular operations. nvMolKit links to RDKit and nvMolKit operations work on RDKit RDMol objects.

nvMolKit mimics RDKit's API where possible, but provides batch-oriented versions of these operations to enable efficient parallel processing of multiple molecules on the GPU.

For operations that don't modify RDKit structures, nvMolKit returns asynchronous GPU results, which can be converted to torch Tensors or numpy arrays. See the :ref:`async-results` section for more details.

An example using nvMolKit to compute Morgan fingerprints in parallel on the GPU is shown below:

.. code-block:: python

    # RDKit API as common base
    from rdkit import Chem
    mols = [Chem.MolFromSmiles(smi) for smi in ['C1CCCCC1', 'C1CCCCC2CCCCC12', "COO"]]

    # Fingerprints via RDKit
    from rdkit.Chem import rdFingerprintGenerator
    rdkit_fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    rdkit_fps = [rdkit_fpgen.GetFingerprint(mol) for mol in mols]  # Sequential processing, list of RDKit fingerprints

    # Fingerprints via nvMolKit
    from nvmolkit.fingerprints import MorganFingerprintGenerator
    nvmolkit_fpgen = MorganFingerprintGenerator(radius=2, fpSize=1024)
    nvmolkit_fps = nvmolkit_fpgen.GetFingerprints(mols)  # Parallel GPU processing, matrix with each row being a fingerprint
    torch.cuda.synchronize()
    print(nvmolkit_fps.torch())

For APIs that modify RDKit structures, nvMolKit applies changes in-place as is done with RDKit. An example of conformer generation:

.. code-block:: python

    from rdkit.Chem.rdDistGeom import ETKDGv3
    from rdkit.Chem import MolFromSmiles, AddHs
    from nvmolkit.embedMolecules import EmbedMolecules as nvMolKitEmbed

    # ETKDG conformer generation via nvMolKit
    mols = [AddHs(MolFromSmiles(smi)) for smi in ['C1CCCCC1', 'C1CCCCC2CCCCC12', "COO"]]
    params = ETKDGv3()
    params.useRandomCoords = True  # Required for nvMolKit
    nvMolKitEmbed(
        molecules=mols,
        params=params,
        confsPerMolecule=5,
        maxIterations=-1,  # Automatic iteration calculation
    )
    for mol in mols:
        print(mol.GetNumConformers())


For more fully-fledged examples, check out the Jupyter notebooks in the `examples` folder of the repository.

Installation
------------

See installation instructions in the GitHub README.


Features
--------

nvMolKit currently supports the following features:

* **Morgan Fingerprints**: Generate Morgan fingerprints for batches of molecules in parallel on GPU
    * Supports fingerprint sizes 128, 256, 512, 1024, and 2048 bits

* **Molecular Similarity**: Fast GPU-accelerated similarity calculations
    * Tanimoto and cosine Similarity
    * Supports all-to-all comparisons between fingerprints in a batch or between two batches of fingerprints

* **ETKDG Conformer Generation**: GPU-accelerated 3D conformer generation using Experimental-Torsion Knowledge-based Distance Geometry
    * Batch processing of multiple molecules with multiple conformers per molecule

* **MMFF Geometry Relaxation**: GPU-accelerated molecular mechanics force field optimization
    * MMFF94 force field implementation for conformer optimization
    * Batch optimization of multiple molecules and conformers


.. _async-results:

Asynchronous GPU Results
------------------------

nvMolKit operations that return GPU-resident data (such as fingerprinting or similarity) return an ``AsyncGpuResult`` object.
This object wraps a GPU computation and allows you to access the results in different formats.

.. code-block:: python

    # Example using fingerprints
    from nvmolkit.fingerprints import MorganFingerprintGenerator
    fpgen = MorganFingerprintGenerator(radius=2, fpSize=1024)
    
    # Get fingerprints - returns AsyncGpuResult. This can be passed to other functions that accept AsyncGpuResult such as similarity.
    # It can be passed to other nvMolKit functions without synchronization, so that multiple operations on the same GPU can be queued
    # before the first one finishes
    result = fpgen.GetFingerprints(mols)

    # To access a result, first synchronize then convert to desired format
    torch.cuda.synchronize()
    
    # Convert to torch tensor (stays on GPU, zero copy)
    fps_torch = result.torch()
    
    # Convert to numpy array (moves to CPU)
    fps_numpy = result.numpy()

The "asynchronous" nature of nvMolKit operations allows you to queue multiple GPU operations without waiting for each to complete. 
You can then choose when to synchronize with the GPU and retrieve results or launch additional operations. Numpy conversions involve
synchronizing with the GPU before copy to the GPU. For torch operations, synchronization can be achieved at any time via `torch.cuda.synchronize()`.

NOTE: Support for streams is not yet implemented, all operations are executed on the default stream.


Hardware targeting
------------------

Some operations (currently conformer generation and energy relaxation) support multiple GPUs,and have options for
controlling tunable performance parameters. The ``BatchHardwareOptions`` class can be used to specify these options.

An example:

.. code-block:: python

    from nvmolkit.types import HardwareOptions
    from nvmolkit.embedMolecules import EmbedMolecules

    options = HardwareOptions()

    # Target GPUs 0, 1 and 2. Defaults to using all GPUs detected
    options.gpuIds = [0, 1, 2]

    # Use 12 threads for parallel preprocessing. Defaults to using all CPUs detected
    options.preprocessingThreads = 12

    # Divide up the work into batches of 500 conformers at a time. nvMolKit will pick a reasonable default but
    # optimal values may depend on the GPU.
    options.batchSize = 500

    # Process 4 batches per GPU in parallel
    options.batchesPerGpu = 4
    EmbedMolecules(mols, ..., hardwareOptions=options)



API Reference
-------------

.. toctree::
   :maxdepth: 1

   api/nvmolkit

What's New
----------

.. toctree::
   :maxdepth: 1

   changelog
