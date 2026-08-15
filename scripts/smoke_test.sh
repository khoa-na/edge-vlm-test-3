#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY="${PYTHON_BIN}"
elif [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY="python3"
else
  PY="python"
fi

echo "Python: $(${PY} --version)"
bash -n scripts/setup_models.sh scripts/smoke_test.sh
${PY} -m pytest tests/ -q
${PY} -m src.demo >/tmp/edge_vlm_demo_smoke.log
echo "Demo: OK (log: /tmp/edge_vlm_demo_smoke.log)"
${PY} -m src.red_team >/tmp/edge_vlm_red_team.log
tail -n 1 /tmp/edge_vlm_red_team.log
echo "Red-team: OK (log: /tmp/edge_vlm_red_team.log)"

if ./scripts/setup_models.sh check >/tmp/edge_vlm_model_check.log 2>&1; then
  echo "Models: all present"
else
  echo "Models: optional files missing; see /tmp/edge_vlm_model_check.log"
fi
