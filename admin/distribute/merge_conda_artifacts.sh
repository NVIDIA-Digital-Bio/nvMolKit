#!/usr/bin/env bash
#
# Merge multiple conda build_artifacts directories into one channel directory
# so that verify_conda_version_combinations.sh can use a single local endpoint.
#
# Conda does NOT search recursively: it expects one channel root with subdirs
# like linux-64/, noarch/. This script copies every .conda and .tar.bz2 from
# each build_artifacts/<subdir>/ into a single merged directory.
#
# Usage (after unzip_all_conda_artifacts.sh - two args only):
#   $0 <folder_containing_unzipped_artifacts> <merged_output_dir>
#
#   First arg = folder that has zip files and unzipped folders; each unzipped
#   folder contains a build_artifacts/ dir. Script finds all build_artifacts and
#   merges them into merged_output_dir (created if needed).
#
# Example:
#   ./merge_conda_artifacts.sh /path/to/conda_zip /path/to/merged_channel
#

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <folder_containing_unzipped_artifacts> <merged_output_dir>" >&2
  echo "" >&2
  echo "Run after unzip_all_conda_artifacts.sh. Merges all build_artifacts into the second folder." >&2
  exit 1
fi

SOURCE_FOLDER=$1
MERGED_DIR=$2

if [[ ! -d "$SOURCE_FOLDER" ]]; then
  echo "Source folder does not exist: $SOURCE_FOLDER" >&2
  exit 1
fi

SOURCE_FOLDER=$(cd "$SOURCE_FOLDER" && pwd)
mkdir -p "$MERGED_DIR"
MERGED_DIR=$(cd "$MERGED_DIR" && pwd)

BUILD_ARTIFACTS=()
while IFS= read -r -d '' d; do
  BUILD_ARTIFACTS+=("$d")
done < <(find "$SOURCE_FOLDER" -type d -name "build_artifacts" -print0 2>/dev/null | sort -z)

if [[ ${#BUILD_ARTIFACTS[@]} -eq 0 ]]; then
  echo "No build_artifacts directories found under $SOURCE_FOLDER" >&2
  echo "Expected structure: $SOURCE_FOLDER/<unzipped_folder>/build_artifacts/linux-64/..." >&2
  exit 1
fi

echo "Found ${#BUILD_ARTIFACTS[@]} build_artifacts folder(s)"

for path in "${BUILD_ARTIFACTS[@]}"; do
  if [[ ! -d "$path" ]]; then
    echo "Skipping (not a directory): $path" >&2
    continue
  fi
  for subdir in linux-64 linux-aarch64 osx-64 osx-arm64 win-64 noarch; do
    if [[ -d "$path/$subdir" ]]; then
      mkdir -p "$MERGED_DIR/$subdir"
      for f in "$path/$subdir"/*.conda "$path/$subdir"/*.tar.bz2; do
        [[ -e $f ]] || continue
        cp "$f" "$MERGED_DIR/$subdir/"
      done
    fi
  done
done

echo "Merged into $MERGED_DIR."
if command -v conda-index >/dev/null 2>&1; then
  conda index "$MERGED_DIR" && echo "Indexed (repodata generated)."
elif python -m conda_index --help >/dev/null 2>&1; then
  python -m conda_index "$MERGED_DIR" && echo "Indexed (repodata generated)."
else
  echo "Tip: install conda-index for repodata (optional): conda install conda-index"
fi
echo "Done. Use as local_conda_endpoint: ./verify_conda_version_combinations.sh $MERGED_DIR <pytest_dir> <log_dir>"
