# Changelog

## 0.4.0 - 2026-02-23

### Summary

nvMolKit 0.4.0 adds GPU-accelerated substructure searching, optional stream control across Python APIs, and enhancements to Butina clustering.

### Contributors
- Kevin Boyd (@scal444)
- Eva Xue (@evasnow1992)

### Features
- GPU-accelerated substructure search with `hasSubstructMatch`, `countSubstructMatches`, and `getSubstructMatches`. Supports batch queries against batch targets with SMARTS-based query molecules.
- Optional `stream` parameter added to fingerprint generation, similarity, and Butina clustering APIs, enabling explicit CUDA stream control
- Butina clustering now supports optional centroid reporting via the `return_centroids` parameter ([#82](https://github.com/NVIDIA-Digital-Bio/nvMolKit/pull/82))
- Butina clustering performance improved by replacing CPU loops with CUDA Graph conditional nodes ([#72](https://github.com/NVIDIA-Digital-Bio/nvMolKit/pull/72))

### Bug Fixes
- Fix data races when torch operations immediately followed nvMolKit calls on the default stream (Issue [#84](https://github.com/NVIDIA-Digital-Bio/nvMolKit/issues/84)). Operations now correctly use the current stream or an explicit `stream` parameter ([#36](https://github.com/NVIDIA-Digital-Bio/nvMolKit/issues/36)).
- Fix `setup.py` compatibility on some Python versions and rework CUDA target detection ([#68](https://github.com/NVIDIA-Digital-Bio/nvMolKit/pull/68))

## 0.3.0 - 2025-12-12

### Summary

nvMolKit 0.3.0 adds Butina clustering support, improved performance to MMFF relaxation and conformer generation, and increased compatibility with libraries and compilers.

### Contributors
- Kevin Boyd (@scal444)
- Eva Xue (@evasnow1992)
- Xuangui Huang (@stslxg-nv)

### Features
- Butina clustering API enabled, using distance matrix input. On an H200 GPU, speedups of 400-1000x can be achieved on datasets up to 60k molecules
- Improvements to BFGS minimizer. Up to 5x speedup compared to nvMolKit v0.2 on batches of small molecules (<20 atoms), with ~10-20% speedup in the general case. Applies to both MMFF relaxation and conformer generation.
- Conda-forge releases now support RDKit versions 2024.9.6 to 2025.9.3

### Bug Fixes
- Fixed a bug where synchronizations on the wrong stream could lead to data races in tests (Issue #28)
- Fixed several areas where a memcpy could go out of scope before completing (Issue #28, Issue #29)
- Fixed a bug where ETKDG would exit early with small CPU counts due to an incorrect identification of resource mis-configuration (Issue #31)

### Miscellaneous
- (C++) Added support for CUB/CCCL > v2.8
- (C++) Added support for externally specified CCCL
- (C++) Added support for CUDA 13.0

## 0.2.0 - 2025-10-24

### Summary

nvMolKit 0.2.0 comes with significant usability and feature-completeness improvements to existing functionality. It is also
the first release to have a [conda-forge release](https://anaconda.org/conda-forge/nvmolkit).

### Contributors
- Kevin Boyd (@scal444)
- Eva Xue (@evasnow1992)
- Ignacio Pickering (@ignaciojpickering)

### Features
- Add memory-segmented cross-similarity code, enabling larger datasets on systems with limited GPU memory ([#13](https://github.com/NVIDIA-Digital-Bio/nvMolKit/pull/13))
- Support conformer deduplication in ETKDG conformer generation ([#14](https://github.com/NVIDIA-Digital-Bio/nvMolKit/pull/14))
- Allow molecules > 256 atoms in conformer generation and MMFF optimization ([#16](https://github.com/NVIDIA-Digital-Bio/nvMolKit/pull/16))
- Enable all combinations of (ET)(K)(DG) in conformer generator ([#17](https://github.com/NVIDIA-Digital-Bio/nvMolKit/pull/17))

### Bug Fixes
- Fix compilation error on C++ build with target=native on Hopper architecture GPUs. ([#6](https://github.com/NVIDIA-Digital-Bio/nvMolKit/pull/6))
- Fix lack of device-set cleanup in multi-GPU code ([#8](https://github.com/NVIDIA-Digital-Bio/nvMolKit/pull/8))
- Fix bug in fingerprint bool->bitfield packing/unpacking code ([#11](https://github.com/NVIDIA-Digital-Bio/nvMolKit/pull/11))
- Fix integer overflow leading to incorrect allocations in similarity calculation code. ([#20](https://github.com/NVIDIA-Digital-Bio/nvMolKit/pull/20))
- Fix crash in most multithreaded APIs whenever exceptions are thrown inside of OpenMP loop. Exceptions now properly propagated to python ([#18](https://github.com/NVIDIA-Digital-Bio/nvMolKit/pull/18))

### Miscellaneous
- Removed unsupported Bulk Similarity APIs ([#12](https://github.com/NVIDIA-Digital-Bio/nvMolKit/pull/12))

## 0.0.1  2025-09-09

### Summary

Initial release of nvMolKit. Features include:
- Morgan Fingerprints
- Tanimoto and Cosine similarity
- ETKDG conformer generation
- MMFF optimization
