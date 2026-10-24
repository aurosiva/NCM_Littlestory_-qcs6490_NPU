# Source me: QAIRT 2.50 host toolchain (python 3.10, numpy 1.26.4, onnx 1.16.1).
# Set these for your machine before sourcing, e.g.
#   export QAIRT_SDK_ROOT=$HOME/qairt/2.50.0.260828
#   export QAIRT_PY_ENV=$HOME/miniconda3/envs/qairt
: "${QAIRT_SDK_ROOT:?set QAIRT_SDK_ROOT to your QAIRT 2.50 SDK folder}"
[ -n "${QAIRT_PY_ENV:-}" ] && export PATH="$QAIRT_PY_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1          # keep ~/.local numpy 2.x out (it breaks the SDK's native modules)
export TMPDIR="${TMPDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.smoke/tmp}"
mkdir -p "$TMPDIR"
source "$QAIRT_SDK_ROOT/bin/envsetup.sh" >/dev/null
