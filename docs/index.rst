.. SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

NVIDIA nvMolKit Documentation
===================================

nvMolKit Introduction
-----------------

Welcome to the NVIDIA nvMolKit documentation!

Installation
------------

<UNDER CONSTRUCTION>

Overview
--------

nvMolKit is a CUDA-backed python library for accelerating common RDKit molecular operations. nvMolKit links to RDKit and nvMolKit operations work on RDKit RDMol objects. 

nvMolKit mimics RDKit's API where possible, but provides batch-oriented versions of these operations to enable efficient parallel processing of multiple molecules on the GPU. 

nvMolKit returns asynchronous GPU results, which can be converted to torch Tensors or numpy arrays. See the :ref:`async-results` section for more details.


Features
--------

nvMolKit currently supports the following features:

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

nvMolKit operations that run on the GPU return an ``AsyncGpuResult`` object. This object wraps a GPU computation and allows you to access the results in different formats.

The "asynchronous" nature of nvMolKit operations allows you to queue multiple GPU operations without waiting for each to complete. 
You can then choose when to synchronize with the GPU and retrieve results or launch additional operations. Numpy conversions involve
synchronizing with the GPU before copy to the GPU. For torch operations, synchronization can be achieved at any time via `torch.cuda.synchronize()`.

NOTE: Support for streams is not yet implemented, all operations are executed on the default stream.


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
