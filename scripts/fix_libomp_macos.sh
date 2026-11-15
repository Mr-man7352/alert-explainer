#!/usr/bin/env bash
# Make LightGBM find an OpenMP runtime (libomp.dylib) of the right CPU architecture on macOS,
# without Homebrew. It copies the libomp that scikit-learn's wheel already bundles (same
# architecture as the project's Python) into the Python's lib/ folder, which LightGBM searches.
#
# Use this when `import lightgbm` fails with "Library not loaded: @rpath/libomp.dylib",
# e.g. on an Apple Silicon Mac whose Homebrew is the old Intel one in /usr/local.
# Usage: scripts/fix_libomp_macos.sh
set -euo pipefail
[[ "$(uname)" == "Darwin" ]] || { echo "Only needed on macOS."; exit 0; }

if uv run python -c "import lightgbm" 2>/dev/null; then
  echo "LightGBM already loads; nothing to do."; exit 0
fi

py_arch=$(uv run python -c "import platform; print(platform.machine())")
py_lib=$(uv run python -c "import os, sys; print(os.path.join(sys.base_prefix, 'lib'))")
src=$(uv run python -c "import os, sklearn; print(os.path.join(os.path.dirname(sklearn.__file__), '.dylibs', 'libomp.dylib'))")
[[ -f "$src" ]] || { echo "scikit-learn's bundled libomp not found at $src"; exit 1; }

echo "Python architecture: $py_arch"
echo "Copying $src"
echo "     to $py_lib/libomp.dylib"
cp "$src" "$py_lib/libomp.dylib"

uv run python -c "import lightgbm; print('LightGBM', lightgbm.__version__, 'loads OK')"
