#!/usr/bin/env bash
# Clone BoolODE at the pinned commit and apply our patches (replicate seeds, sklearn>=1.0 fix, optional
# time-point thinning + rounding used by the tree-depth experiment). Run from anywhere:  bash scripts/setup_boolode.sh
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -d external/BoolODE/.git ]]; then echo "external/BoolODE exists already"; exit 0; fi
rm -rf external/BoolODE
git clone https://github.com/Murali-group/BoolODE external/BoolODE
git -C external/BoolODE checkout "$(cat external/boolode_pin/commit.txt)"
git -C external/BoolODE apply ../boolode_pin/patches.diff
echo "BoolODE ready at external/BoolODE ($(cat external/boolode_pin/commit.txt), patched)"
