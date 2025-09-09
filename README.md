# nvMolKit

## Documentation
Please see the official [NVIDIA nvMolKit Documentation](https://nvidia-digital-bio.github.io/nvMolKit/) for an overview of features, examples, and a detailed API reference.

## Installation Guide

### Prerequisites

#### An NVIDIA GPU
nvMolKit requires an NVIDIA GPU with compute capability 7.0 (V100) or higher. Check your GPU's compute capability [here](https://developer.nvidia.com/cuda-gpus).
Multiple GPUs are supported for some APIs.

#### System Dependencies

First, install essential build dependencies. This includes a C++ compiler and OpenMP. Eigen headers may be necesary,
sometimes RDKit includes them in some headers but the RDKit install does not always properly declare this dependency.

Example shown for installing on Ubuntu. System installs or conda installs both work.
```bash
# Update package list
sudo apt-get update

# Install build tools and development headers
sudo apt-get install build-essential libeigen3-dev
sudo apt-get install libstdc++-12-dev libomp-15-dev

# nvMolKit requires a C++ compiler. You can install it system-wide or via conda:

# Example: Install clang on Ubuntu:
sudo apt-get install clang-15 clang-format-15 clang-tidy-15

# Other options:
# - Use system GCC (already included in build-essential above)
# - Install inside a conda environment (see Python Environment Setup section below):
#   conda install -c conda-forge cxx-compiler
```

#### CUDA Installation

Install NVIDIA CUDA Toolkit (version 12.5 or later) following [NVIDIA's official installation guide](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_network).

#### CMake

nvMolKit requires CMake >= 3.26. Update if needed, for example on Ubuntu:

```bash
# Remove old CMake
sudo apt remove --purge --auto-remove cmake

# Install CMake 3.30.1
wget https://github.com/Kitware/CMake/releases/download/v3.30.1/cmake-3.30.1-linux-x86_64.sh
chmod +x cmake-3.30.1-linux-x86_64.sh
sudo ./cmake-3.30.1-linux-x86_64.sh --prefix=/usr/local --skip-license

# Verify installation
cmake --version
```

A conda cmake should also work.


#### Python Environment Setup

Create a conda environment with all required dependencies (install [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) or [Anaconda](https://www.anaconda.com/download) if you don't have conda):

```bash
# Create and activate environment
conda create --name nvmolkit_dev_py312 python=3.12.1
conda activate nvmolkit_dev_py312

# Install RDKit with development headers
conda install -c conda-forge rdkit=2024.09.3 rdkit-dev=2024.09.3

# Install Boost subpackages in case RDKit install did not include them transitively
conda install -c conda-forge libboost libboost-python libboost-devel libboost-headers libboost-python-devel

# Install Torch, make sure it's a GPU-enabled version. If having trouble install, check out the
# torch installation guidelines: https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio
python -c "import torch; print(torch.__version__); print(f'Is a CUDA build? {torch.cuda.is_available()}')"
```

### Installation

```bash
# Activate your environment
conda activate nvmolkit_dev_py312

# Navigate to the repo root directory
cd <path/to/nvmolkit>

# Install nvMolKit directly
#  Use all CPU cores for a faster build, or replace $(nproc) with a specific number
CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) pip -v install .
```
This will build and install nvMolKit with Python bindings automatically.

To test the installation, run
```bash
pip install pytest
(cd nvmolkit/tests && pytest -v .)
```

### Docker containers

Materials for building docker containers are in [`admin/container`](admin/container). [HPCCM](https://github.com/NVIDIA/hpc-container-maker)
builds docker files from yaml configs. See the [README](admin/container/hpccm_build.py) for config definitions.


## Developer Guide

### cppcheck Installation

For code quality analysis during development:

```bash
wget https://github.com/danmar/cppcheck/archive/2.14.2.tar.gz
tar -zxvf 2.14.2.tar.gz
cd cppcheck-2.14.2
mkdir build && cd build
cmake .. -DUSE_MATCHCOMPILER=ON -DCMAKE_BUILD_TYPE=release
make -j
sudo make install
cd ../../ && rm -rf cppcheck-2.14.2 2.14.2.tar.gz
```

### Development Build Options

#### Compilers

- **Supported Compilers**: We have tested and support clang-15 and GCC 12. Other compilers may work but are not extensively tested.

#### CMake Build Options

- **`-DCMAKE_BUILD_TYPE=<type>`**: Available options for build type include `Release`, `Debug`, `RelWithDebInfo`, `asan`, `tsan`, and `ubsan`.

- **`-DNVMOLKIT_BUILD_TESTS=ON`**: Enables building unit tests. CMake will download and build GTest automatically. Run tests with `ctest` after building.

- **`-DNVMOLKIT_BUILD_BENCHMARKS=ON`**: Enables building performance benchmarks. CMake will download and build nanobench automatically. After building, executable benchmarks will be found in `build/benchmarks`.

- **`-DNVMOLKIT_BUILD_PYTHON_BINDINGS=ON`**: Builds Python bindings using boost-python. Required for Python API access. This ensures compatibility with RDKit's Python bindings.

- **`-DNVMOLKIT_CUDA_TARGET_MODE=<mode>`**: Controls GPU target architectures. See GPU Target Architectures section below for available modes.

- **`-DNVMOLKIT_BUILD_AGAINST_PIP_RDKIT=ON`**: Build against pip-installed RDKit instead of conda. See Building Against pip-installed RDKit section below for additional required configuration. Default: `OFF`.

#### GPU Target Architectures

nvMolKit supports building for multiple GPU architectures. Build behavior is controlled by the `NVMOLKIT_CUDA_TARGET_MODE` variable:

- **`default`**: Uses `CMAKE_CUDA_ARCHITECTURES` if set, otherwise defaults to compute capability 7.0
- **`native`**: Builds only for the GPU on your current system. Fastest for local development but not portable.
- **`full`**: Builds for all architectures >= 7.0, including Blackwell (if NVCC >= 12.8). Larger binaries, longer compile time, but works on all major GPUs.

**Recommendation**: Use `native` for development, `full` for distribution.

#### Building Against pip-installed RDKit

**Note**: The conda-based setup above is strongly recommended. This section is for advanced users only.

If you must build against pip-installed RDKit (not recommended), you'll need to manually provide headers since pip packages don't include them. This requires downloading RDKit source code and boost headers separately:

```bash
# Set environment variables for pip install
NVMOLKIT_BUILD_AGAINST_PIP=ON \
NVMOLKIT_BUILD_AGAINST_PIP_LIBDIR=<path to rdkit.libs> \
NVMOLKIT_BUILD_AGAINST_PIP_INCDIR=<path to rdkit headers> \
NVMOLKIT_BUILD_AGAINST_PIP_BOOSTINCLUDEDIR=<path to boost headers> \
pip install .
```

This approach is error-prone. We recommend using the conda environment setup instead.

