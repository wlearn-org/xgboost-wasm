#!/bin/bash
set -euo pipefail

# Verify all required C API symbols are present in the Emscripten glue
GLUE="${1:-wasm/xgboost.js}"

if [ ! -f "$GLUE" ]; then
  echo "ERROR: Glue file not found: $GLUE"
  exit 1
fi

REQUIRED_SYMBOLS=(
  XGBGetLastError
  XGDMatrixCreateFromMat
  XGDMatrixSetFloatInfo
  XGDMatrixNumRow
  XGDMatrixNumCol
  XGDMatrixFree
  XGBoosterCreate
  XGBoosterSetParam
  XGBoosterUpdateOneIter
  XGBoosterEvalOneIter
  XGBoosterPredictFromDMatrix
  XGBoosterSaveModelToBuffer
  XGBoosterLoadModelFromBuffer
  XGBoosterFree
)

MISSING=0
for fn in "${REQUIRED_SYMBOLS[@]}"; do
  if ! grep -q "_${fn}" "$GLUE"; then
    echo "MISSING: ${fn}"
    MISSING=$((MISSING + 1))
  fi
done

if [ $MISSING -gt 0 ]; then
  echo "ERROR: ${MISSING} required symbol(s) missing from ${GLUE}"
  exit 1
fi

echo "All ${#REQUIRED_SYMBOLS[@]} exports verified in ${GLUE}"
