#!/usr/bin/env bash
#
# Unzip all .zip files in a given folder (e.g. Azure conda artifact zips).
# Handles double-zip layout: first unzips top-level zips, then unzips any .zip
# found inside the resulting directories (so build_artifacts/ ends up visible).
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
  echo "Unzips every .zip in the folder, then unzips any .zip inside resulting dirs (two rounds)." >&2
  exit 1
fi

FOLDER=$1
if [[ ! -d "$FOLDER" ]]; then
  echo "Folder does not exist: $FOLDER" >&2
  exit 1
fi

FOLDER=$(cd "$FOLDER" && pwd)
count=0

# Round 1: unzip all .zip in the top-level folder
for zip in "$FOLDER"/*.zip; do
  [[ -f "$zip" ]] || continue
  echo "Unzipping (round 1): $zip"
  unzip -o -q "$zip" -d "$FOLDER"
  (( count++ )) || true
done

# Round 2: find any .zip inside subdirs and unzip into that same subdir
while IFS= read -r -d '' zip; do
  dir=$(dirname "$zip")
  echo "Unzipping (round 2): $zip"
  unzip -o -q "$zip" -d "$dir"
  (( count++ )) || true
done < <(find "$FOLDER" -mindepth 1 -type f -name "*.zip" -print0 2>/dev/null)

if [[ $count -eq 0 ]]; then
  echo "No .zip files found in $FOLDER" >&2
  exit 1
fi

echo "Done. Unzipped $count file(s). Next: run merge_conda_artifacts.sh $FOLDER <merged_output_dir>"
