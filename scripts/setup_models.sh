#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-tier1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_DIR="${REPO_ROOT}/models"
mkdir -p "${MODEL_DIR}"

download_checked() {
  local url="$1"
  local destination="$2"
  local expected_sha="$3"
  if [[ -f "${destination}" ]] && echo "${expected_sha}  ${destination}" | sha256sum -c --status; then
    echo "OK  ${destination#"${REPO_ROOT}/"}"
    return
  fi
  local partial="${destination}.part"
  echo "GET ${url}"
  curl -fL --retry 3 --continue-at - --output "${partial}" "${url}"
  echo "${expected_sha}  ${partial}" | sha256sum -c --status
  mv "${partial}" "${destination}"
  echo "OK  ${destination#"${REPO_ROOT}/"}"
}

setup_tier1() {
  download_checked \
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" \
    "${MODEL_DIR}/face_landmarker.task" \
    "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
  download_checked \
    "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt" \
    "${MODEL_DIR}/yolo26n.pt" \
    "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"
}

setup_vlm() {
  echo "Tier 2 cần khoảng 2 GB tải xuống. Các file GGUF không được commit vào Git."
  download_checked \
    "https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_K_M.gguf" \
    "${MODEL_DIR}/qwen3.5-2b-q4_k_m.gguf" \
    "aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223"
  download_checked \
    "https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/mmproj-F16.gguf" \
    "${MODEL_DIR}/qwen3.5-2b-mmproj-f16.gguf" \
    "7035e9cb8d7c6a9681d07eef9a364783e86ea4cd73faab2eabb4f43a101830c7"
}

case "${MODE}" in
  tier1)
    setup_tier1
    ;;
  vlm)
    setup_vlm
    ;;
  all)
    setup_tier1
    setup_vlm
    ;;
  check)
    status=0
    for f in face_landmarker.task yolo26n.pt qwen3.5-2b-q4_k_m.gguf qwen3.5-2b-mmproj-f16.gguf; do
      if [[ -f "${MODEL_DIR}/${f}" ]]; then
        echo "FOUND models/${f} ($(du -h "${MODEL_DIR}/${f}" | cut -f1))"
      else
        echo "MISS  models/${f}"
        status=1
      fi
    done
    exit "${status}"
    ;;
  *)
    echo "Usage: $0 {tier1|vlm|all|check}" >&2
    exit 2
    ;;
esac
