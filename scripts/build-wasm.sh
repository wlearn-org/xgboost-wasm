#!/bin/bash
set -euo pipefail

# Build XGBoost v3.2.0 as WASM via Emscripten
# Prerequisites: emsdk activated (emcc, emcmake, emmake in PATH)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
UPSTREAM_DIR="${PROJECT_DIR}/upstream/xgboost"
BUILD_DIR="${PROJECT_DIR}/build"
OUTPUT_DIR="${PROJECT_DIR}/wasm"

# Verify prerequisites
if ! command -v emcc &> /dev/null; then
  echo "ERROR: emcc not found. Activate emsdk first:"
  echo "  source /path/to/emsdk/emsdk_env.sh"
  exit 1
fi

if [ ! -f "$UPSTREAM_DIR/CMakeLists.txt" ]; then
  echo "ERROR: XGBoost upstream not found at ${UPSTREAM_DIR}"
  echo "  git submodule update --init --recursive"
  exit 1
fi

echo "=== Applying patches ==="
if [ -d "${PROJECT_DIR}/patches" ] && ls "${PROJECT_DIR}/patches"/*.patch &> /dev/null; then
  for patch in "${PROJECT_DIR}/patches"/*.patch; do
    echo "Applying: $(basename "$patch")"
    (cd "$UPSTREAM_DIR" && git apply --check "$patch" 2>/dev/null && git apply "$patch") || \
      echo "  (already applied or not applicable)"
  done
else
  echo "  No patches found"
fi

echo "=== CMake configure ==="
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

emcmake cmake "$UPSTREAM_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_CUDA=OFF \
  -DUSE_NCCL=OFF \
  -DUSE_OPENMP=OFF \
  -DPLUGIN_FEDERATED=OFF \
  -DPLUGIN_RMM=OFF \
  -DBUILD_STATIC_LIB=ON \
  -DCMAKE_C_FLAGS="-O2 -DDMLC_LOG_STACK_TRACE=0 -fexceptions" \
  -DCMAKE_CXX_FLAGS="-O2 -DDMLC_LOG_STACK_TRACE=0 -fexceptions"

echo "=== Building ==="
emmake make -j"$(nproc)" 2>&1

echo "=== Linking WASM ==="
mkdir -p "$OUTPUT_DIR"

# Find the static library — may be in build/ or upstream/lib/
XGBOOST_LIB=$(find "$BUILD_DIR" "$UPSTREAM_DIR" -name 'libxgboost.a' -print -quit)
DMLC_LIB=$(find "$BUILD_DIR" "$UPSTREAM_DIR" -name 'libdmlc.a' -print -quit)

if [ -z "$XGBOOST_LIB" ]; then
  echo "ERROR: libxgboost.a not found in build directory"
  find "$BUILD_DIR" -name '*.a' -print
  exit 1
fi

echo "Using: $XGBOOST_LIB"
[ -n "$DMLC_LIB" ] && echo "Using: $DMLC_LIB"

LINK_LIBS="-Wl,--whole-archive $XGBOOST_LIB -Wl,--no-whole-archive"
[ -n "$DMLC_LIB" ] && LINK_LIBS="$LINK_LIBS $DMLC_LIB"

emcc $LINK_LIBS \
  -o "${OUTPUT_DIR}/xgboost.js" \
  -s MODULARIZE=1 \
  -s EXPORT_NAME=createXGBoost \
  -s EXPORTED_FUNCTIONS='["_XGBGetLastError","_XGDMatrixCreateFromMat","_XGDMatrixSetFloatInfo","_XGDMatrixNumRow","_XGDMatrixNumCol","_XGDMatrixFree","_XGBoosterCreate","_XGBoosterSetParam","_XGBoosterUpdateOneIter","_XGBoosterEvalOneIter","_XGBoosterPredictFromDMatrix","_XGBoosterSaveModelToBuffer","_XGBoosterLoadModelFromBuffer","_XGBoosterFree","_malloc","_free"]' \
  -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap","getValue","setValue","HEAPF32","HEAPU8","UTF8ToString"]' \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s MAXIMUM_MEMORY=2147483648 \
  -s ENVIRONMENT='web,node' \
  -s SINGLE_FILE=1 \
  -fexceptions \
  -O2

echo "=== Verifying exports ==="
bash "${SCRIPT_DIR}/verify-exports.sh" "${OUTPUT_DIR}/xgboost.js"

echo "=== Writing BUILD_INFO ==="
cat > "${OUTPUT_DIR}/BUILD_INFO" <<EOF
upstream: xgboost v3.2.0
upstream_commit: $(cd "$UPSTREAM_DIR" && git rev-parse HEAD)
build_date: $(date -u +%Y-%m-%dT%H:%M:%SZ)
emscripten: $(emcc --version | head -1)
build_flags: -O2 -DDMLC_LOG_STACK_TRACE=0 SINGLE_FILE=1
wasm_embedded: true
EOF

echo "=== Build complete ==="
ls -lh "${OUTPUT_DIR}/xgboost.js"
cat "${OUTPUT_DIR}/BUILD_INFO"
