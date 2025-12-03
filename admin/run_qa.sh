#!/bin/bash

source /usr/local/anaconda/etc/profile.d/conda.sh
set -ex
bash admin/run_clang_format.sh -d
bash admin/run_cmake_format_lint.sh -d
bash admin/run_clang_tidy.sh
bash admin/run_cppcheck.sh
bash admin/run_iwyu.sh