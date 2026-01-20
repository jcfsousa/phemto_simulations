#!/bin/bash
set -e

bash ./runCosimaCrabBackground.sh
echo "Cosima finished, starting Revan"
bash ./runRevanCrabBackground.sh
