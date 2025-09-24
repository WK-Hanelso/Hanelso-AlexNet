#!/usr/bin/env bash
set -euo pipefail

# 아키 선택
arch="$(uname -m)"
case "$arch" in
  aarch64|arm64) DEFAULT_IMAGE="u2204_c118_p201-aarch64" ;;
  x86_64|amd64)  DEFAULT_IMAGE="u2204_c118_p230-amd64"  ;;
  *) echo "Unsupported arch: $arch"; exit 1 ;;
esac

# 이미지명(인자로 주면 override)
IMAGE="${1:-$DEFAULT_IMAGE}"
# 컨텍스트(기본 .)
CONTEXT="${2:-.}"

# Dockerfile 자동 선택: ./docker/Dockerfile 우선, 없으면 ./Dockerfile
if [[ -f "./docker/Dockerfile" ]]; then
  DOCKERFILE="./docker/Dockerfile"
elif [[ -f "./Dockerfile" ]]; then
  DOCKERFILE="./Dockerfile"
else
  echo "Dockerfile을 찾을 수 없습니다. ./docker/Dockerfile 또는 ./Dockerfile 위치에 두세요."
  exit 1
fi

# 추가 옵션(원하면 환경변수로)
#   NO_CACHE=true  → --no-cache
#   PULL=true      → --pull
#   PROGRESS=plain → 로그 자세히
EXTRA_ARGS=()
[[ "${NO_CACHE:-false}" == "true" ]] && EXTRA_ARGS+=(--no-cache)
[[ "${PULL:-false}" == "true" ]] && EXTRA_ARGS+=(--pull)
[[ -n "${PROGRESS:-}" ]] && EXTRA_ARGS+=(--progress "$PROGRESS")

export DOCKER_BUILDKIT=${DOCKER_BUILDKIT:-1}

echo "[build] arch=$arch  image=$IMAGE"
echo "[build] Dockerfile: $DOCKERFILE"
echo "[build] Context   : $CONTEXT"
echo "[build] Extra args: ${EXTRA_ARGS[*]:-(none)}"

docker build -t "$IMAGE" -f "$DOCKERFILE" "${EXTRA_ARGS[@]}" "$CONTEXT"

