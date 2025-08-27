# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import dataclasses

import hpccm
import posixpath
import yaml
from hpccm.building_blocks import cmake, conda, generic_cmake, gnu, llvm, packages
from hpccm.primitives import environment, shell
from packaging.version import Version
from hpccm.common import cpu_arch
from hpccm.primitives.comment import comment
from hpccm.primitives.copy import copy

_CONDA_VERSION="25.3.0-3"
@dataclasses.dataclass
class ContainerConfig:
    """Container configuration definition."""
    # Docker image will be built -t <image_name>
    image_name: str
    # Operating system to start with, full path
    image_base: str
    # gcc or clang
    cxx_compiler_id: str
    # Version string, depending on OS it may be the OS-shipped version
    # (e.g. 10, with whatever ubuntu ships in the 10 series)
    cxx_compiler_version: str
    # Which version of CMake to install
    cmake_version: str
    # Python major/minor version
    python_version: str
    python_conda_deps: list[str]
    python_pip_deps: list[str]
    # Whether to install clang-tidy and clang-format
    install_clang_tools: bool

class mamba(conda):
    """Mamba building block, slight alterations from conda, but mostly a copy paste.

    TODO: Upstream changes to allow for mamba in hpccm.
    """

    def __init__(self, **kwargs):
        super(conda, self).__init__(**kwargs)

        self.__arch_pkg = '' # Filled in by __cpu_arch()
        self.__channels = kwargs.get('channels', [])
        self.__environment = kwargs.get('environment', None)

        # By setting this value to True, you agree to the
        # corresponding Anaconda End User License Agreement
        # https://docs.anaconda.com/anaconda/eula/
        self.__eula = kwargs.get('eula', False)
        self.__ospackages = kwargs.get('ospackages',
                                       ['ca-certificates', 'wget'])
        self.__packages = kwargs.get('packages', [])
        self.__prefix = kwargs.get('prefix', '/usr/local/anaconda')
        self.__python2 = kwargs.get('python2', False)
        self.__python_version = '2' if self.__python2 else '3'
        self.__python_subversion = kwargs.get(
            'python_subversion', 'py27' if self.__python2 else 'py312')
        self.__version = kwargs.get('version', '4.8.3' if self.__python2 else '25.1.1-2')

        self.__commands = [] # Filled in by __setup()
        self.__wd = kwargs.get('wd', hpccm.config.g_wd) # working directory

        # Set the CPU architecture specific parameters
        self.__cpu_arch()

        # Construct the series of steps to execute
        self.__setup()

        # Fill in container instructions
        self.__instructions()
    def __setup(self):
        """Construct the series of shell commands, i.e., fill in
           self.__commands"""
        miniconda = f'Miniforge3-{self.__version}-Linux-{self.__arch_pkg}.sh'
        url_prefix = 'https://github.com/conda-forge/miniforge/releases/download'
        url = '{0}/{1}/{2}'.format( url_prefix,self.__version, miniconda)

        # Download source from web
        self.__commands.append(self.download_step(url=url, directory=self.__wd))

        # Install
        install_args = ['-p {}'.format(self.__prefix)]
        if self.__eula:
            install_args.append('-b')
        self.__commands.append('bash {0} {1}'.format(
            posixpath.join(self.__wd, miniconda),
            ' '.join(sorted(install_args))))

        # Initialize conda
        self.__commands.append('{0} init'.format(
            posixpath.join(self.__prefix, 'bin', 'conda')))
        self.__commands.append('ln -s {} /etc/profile.d/conda.sh'.format(
            posixpath.join(self.__prefix, 'etc', 'profile.d', 'conda.sh')))

        # Activate
        if self.__channels or self.__environment or self.__packages:
            self.__commands.append('. {}'.format(
                posixpath.join(self.__prefix, 'etc', 'profile.d', 'conda.sh')))
            self.__commands.append('conda activate base')

        # Enable channels
        if self.__channels:
            self.__commands.append('conda config {}'.format(
                ' '.join(['--add channels {}'.format(x)
                          for x in sorted(self.__channels)])))

        # Install environment
        if self.__environment:
            self.__commands.append('conda env update -f {}'.format(
                posixpath.join(self.__wd,
                               posixpath.basename(self.__environment))))
            self.__commands.append(self.cleanup_step(
                items=[posixpath.join(
                    self.__wd, posixpath.basename(self.__environment))]))

        # Install conda packages
        if self.__packages:
            self.__commands.append('conda install -y {}'.format(
                ' '.join(sorted(self.__packages))))

        # Cleanup conda install
        self.__commands.append('{0} clean -afy'.format(
            posixpath.join(self.__prefix, 'bin', 'conda')))

        # Cleanup miniconda download file
        self.__commands.append(self.cleanup_step(
            items=[posixpath.join(self.__wd, miniconda)]))
    def __instructions(self):
        """Fill in container instructions"""

        self += comment('Anaconda')
        self += packages(ospackages=self.__ospackages)
        if self.__environment:
            self += copy(src=self.__environment, dest=posixpath.join(
                self.__wd, posixpath.basename(self.__environment)))
        self += shell(commands=self.__commands)

    def __cpu_arch(self):
        """Based on the CPU architecture, set values accordingly.  A user
        specified value overrides any defaults."""

        if hpccm.config.g_cpu_arch == cpu_arch.AARCH64:
            self.__arch_pkg = 'aarch64'
        elif hpccm.config.g_cpu_arch == cpu_arch.PPC64LE:
            self.__arch_pkg = 'ppc64le'
        elif hpccm.config.g_cpu_arch == cpu_arch.X86_64:
            self.__arch_pkg = 'x86_64'
        else: # pragma: no cover
            raise RuntimeError('Unknown CPU architecture')



def add_os_packages():
    """Add OS packages required for the container build.

    Returns:
        An HPCCM `packages` building block to install the required OS packages.
    """
    return packages(ospackages=["libeigen3-dev", "git"])

def add_conda_deps(python_version: str, conda_deps: list[str] | None = None):
    """Install Conda with the desired base version and dependencies.

    Args:
        python_version: The Python major/minor version (e.g. '3.9').
        conda_deps: Optional list of additional Conda packages to install.

    Returns:
        An HPCCM `conda` building block to install Conda with the specified version and dependencies.
    """
    if not conda_deps:
        conda_deps = []
    return mamba(
        eula=True,
        version=_CONDA_VERSION,
        channels=['conda-forge', 'nvidia', 'pytorch'],
        packages=[f"python={python_version}", "pip"] + conda_deps
    )

def add_pip_deps(dep_list):
    """Add dependencies to be installed with pip.

    Args:
        dep_list: List of pip dependencies to install.

    Returns:
        An HPCCM `shell` building block to install pip dependencies using shell commands.
    """
    if dep_list:
        return shell(
            commands=[f"/usr/local/anaconda/bin/pip install  {' '.join(dep_list)}"]
        )
    return []

def add_compiler(compiler_id: str, compiler_version: str, install_clang_tools: bool):
    """Add a compiler to the container.

    Args:
        compiler_id: ID of the compiler to add (e.g. 'gcc', 'clang', etc.).
        compiler_version: Version of the compiler to install.
        install_clang_tools: Whether to install Clang tools.

    Returns:
        An HPCCM `gnu` or `llvm` building block to install the specified compiler.

    Raises:
        ValueError: If the compiler ID is invalid or if Clang tools are requested with a non-Clang compiler.
    """
    if compiler_id in ("gcc", "gnu"):
        if install_clang_tools:
            raise ValueError(f"Clang tools requested but compiler is {compiler_id},")
        return gnu(fortran=False, version=compiler_version, extra_repository=True)

    elif compiler_id in ("clang", "llvm"):
        return llvm(upstream=True, version=compiler_version, extra_tools=install_clang_tools, toolset=True)
    else:
        raise ValueError(f"Invalid compiler ID {compiler_id}")


def add_cppcheck():
    """Add CppCheck to the container.

    Returns:
        An HPCCM `generic_cmake` building block to install CppCheck.
    """
    return generic_cmake(
        cmake_opts=["-DUSE_MATCHCOMPILER=ON", "-DCMAKE_BUILD_TYPE=release" ],
        repository="https://github.com/danmar/cppcheck.git",
        commit="fc2210afa95a4cea1afba01d5390d13d6d8d75c8", # 2.14.2
    )



def main():
    """Generate HPC Container Dockerfile.

    This function reads a YAML configuration file, builds an HPC container using hpccm,
    and writes the container specification to a Dockerfile.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    args = parser.parse_args()
    with open(args.config_file, 'r') as f:
        raw_config = yaml.safe_load(f)
    config: ContainerConfig = ContainerConfig(**raw_config)

    stage: hpccm.Stage = hpccm.Stage()
    stage += hpccm.primitives.baseimage(image=config.image_base)
    stage += add_os_packages()
    stage += cmake(eula=True, version=config.cmake_version)
    stage += add_compiler(config.cxx_compiler_id, config.cxx_compiler_version, config.install_clang_tools)
    stage += add_conda_deps(config.python_version, config.python_conda_deps)
    stage += environment(variables={"PATH": "/usr/local/anaconda/bin:$PATH"})
    stage += add_pip_deps(config.python_pip_deps)
    if config.install_clang_tools:
        stage += add_cppcheck()

    with open(f"{config.image_name}.Dockerfile", "w") as f:
        f.write(str(stage))


if __name__ == "__main__":
    main()
