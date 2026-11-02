#!/usr/bin/env bash
# Source this (do NOT execute it):  source activate_phemto_xspec.sh
#
# Initializes the INSTALLED HEASoft/XSPEC and activates the env-phemto venv so
# that `import xspec` works together with numpy/astropy/matplotlib.

export HEADAS="$HOME/apps/heasoft-6.36/x86_64-pc-linux-gnu-libc2.39"
if [ ! -f "$HEADAS/headas-init.sh" ]; then
    echo "activate_phemto_xspec: HEADAS not found at $HEADAS" >&2
    return 1 2>/dev/null || exit 1
fi
source "$HEADAS/headas-init.sh"

VENV="/local/home/jf285468/documents/phd/phemto/phemto_simulations/megalib/analysis_overall/env-phemto"
source "$VENV/bin/activate"

echo "HEASoft + env-phemto ready:"
echo "  HEADAS = $HEADAS"
echo "  python = $(command -v python)"
