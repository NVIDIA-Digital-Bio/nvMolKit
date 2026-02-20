#!/usr/bin/env bash
#
# Unzip all .zip files in a given folder (e.g. Azure conda artifact zips).
# Handles nested zips: unzips top-level .zip files, then recursively unzips any
# .zip found inside the resulting directories (so build_artifacts/ ends up visible).
#
# Usage:
#   $0 <folder_containing_zips>
#
# Example:
#   mkdir -p ~/conda_artifacts_uploads
#   # upload *.zip files there, then:
#   ./unzip_all_conda_artifacts.sh ~/conda_artifacts_uploads
#

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <folder_containing_zips>" >&2
  echo "" >&2
  echo "Unzips every .zip in the folder, then recursively unzips any .zip inside resulting dirs." >&2
  exit 1
fi

FOLDER=$1
if [[ ! -d "$FOLDER" ]]; then
  echo "Folder does not exist: $FOLDER" >&2
  exit 1
fi

FOLDER=$(cd "$FOLDER" && pwd)
count=0

# Top-level: unzip all .zip in the given folder
for zip in "$FOLDER"/*.zip; do
  [[ -f "$zip" ]] || continue
  echo "Unzipping: $zip"
  unzip -o -q "$zip" -d "$FOLDER"
  (( count++ )) || true
done

# Recursive: find any .zip under the tree and unzip into its directory
while IFS= read -r -d '' zip; do
  dir=$(dirname "$zip")
  echo "Unzipping (nested): $zip"
  unzip -o -q "$zip" -d "$dir"
  (( count++ )) || true
done < <(find "$FOLDER" -mindepth 1 -type f -name "*.zip" -print0 2>/dev/null)

if [[ $count -eq 0 ]]; then
  echo "No .zip files found in $FOLDER" >&2
  exit 1
fi

echo "Done. Unzipped $count file(s). Next: run merge_conda_artifacts.sh $FOLDER <merged_output_dir>"
