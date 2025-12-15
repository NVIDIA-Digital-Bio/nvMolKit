.. SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

.. module:: nvmolkit
.. currentmodule:: nvmolkit

nvMolKit APIs
=============


Fingerprint Generation
----------------------

.. autosummary::
   :toctree: generated/
   :template: class_template.rst

   fingerprints.MorganFingerprintGenerator
   fingerprints.pack_fingerprint
   fingerprints.unpack_fingerprint

Similarity Calculations
-----------------------

.. autosummary::
   :toctree: generated/
   :template: function_template.rst

   similarity.crossTanimotoSimilarity
   similarity.crossTanimotoSimilarityMemoryConstrained
   similarity.crossCosineSimilarity
   similarity.crossCosineSimilarityMemoryConstrained


ETKDG Conformer Generation
--------------------------

.. autosummary::
   :toctree: generated/
   :template: function_template.rst

   embedMolecules.EmbedMolecules

MMFF Optimization
-----------------

.. autosummary::
   :toctree: generated/
   :template: function_template.rst

   mmffOptimization.MMFFOptimizeMoleculesConfs

Butina Clustering
-----------------

.. autosummary::
   :toctree: generated/
   :template: function_template.rst

   clustering.butina

Types
-----

.. autosummary::
   :toctree: generated/
   :template: class_template.rst

   types.AsyncGpuResult
   types.HardwareOptions