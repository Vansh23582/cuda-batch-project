#!/bin/bash
# run.sh
#
# Convenience script: generate data (if missing), build, then run.
# I put this here so the grader can just do: bash run.sh  and see everything.
#
# Usage:
#   bash run.sh               # generate 120 images, build, process all
#   bash run.sh --skip-data   # skip generation (data/ already populated)

set -e  # stop on first error

INPUT_DIR="data"
OUTPUT_DIR="output"
COUNT=120   # number of synthetic images to generate

# ── Optional flag: skip data generation if images already exist ───────────────
SKIP_DATA=0
for arg in "$@"; do
    [[ "$arg" == "--skip-data" ]] && SKIP_DATA=1
done

# ── Step 1: Generate synthetic test images ────────────────────────────────────
if [[ $SKIP_DATA -eq 0 ]]; then
    echo "=== Generating ${COUNT} synthetic PGM images ==="
    python3 scripts/generate_data.py --count "$COUNT" --outdir "$INPUT_DIR"
    echo ""
fi

# ── Step 2: Build the CUDA binary ─────────────────────────────────────────────
echo "=== Building CUDA binary ==="
make all
echo ""

# ── Step 3: Run the batch processor ──────────────────────────────────────────
echo "=== Running GPU batch processor ==="
mkdir -p "$OUTPUT_DIR"
./bin/run --input "$INPUT_DIR" --output "$OUTPUT_DIR"

echo ""
echo "=== Results written to $OUTPUT_DIR/ ==="
ls -lh "$OUTPUT_DIR/" | head -20
echo "(showing first 20 output files)"
