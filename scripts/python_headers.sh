#!/usr/bin/env bash
# Source from the project root before starting SGLang. Do not install system packages.
set -euo pipefail
PROJECT_PYTHON=${PROJECT_PYTHON:-"$PWD/.venv/bin/python"}
PYTHON_INCLUDE_VERSION=$("$PROJECT_PYTHON" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])')
PYTHON_INCLUDE_DIR=$("$PROJECT_PYTHON" -c 'import sysconfig; print(sysconfig.get_path("include"))')
if [[ ! -f "$PYTHON_INCLUDE_DIR/Python.h" ]]; then
  # The local headers were extracted from python3.11-devel-3.11.13-2.el8_10.
  if [[ "$PYTHON_INCLUDE_VERSION" != 3.11.13 ]]; then
    echo "Missing Python headers for $PYTHON_INCLUDE_VERSION; install matching development headers or provide them in the project cache" >&2
    return 1 2>/dev/null || exit 1
  fi
  PYTHON_INCLUDE_DIR="$PWD/.cache/python-devel/usr/include/python3.11"
fi
if [[ ! -f "$PYTHON_INCLUDE_DIR/Python.h" ]]; then
  echo "Missing Python development headers at $PYTHON_INCLUDE_DIR" >&2
  return 1 2>/dev/null || exit 1
fi
export CPATH="$PYTHON_INCLUDE_DIR${CPATH:+:$CPATH}"
printf '#include <Python.h>\n' | "${CC:-gcc}" -x c -fsyntax-only - || { echo 'Python header preflight failed' >&2; return 1 2>/dev/null || exit 1; }
