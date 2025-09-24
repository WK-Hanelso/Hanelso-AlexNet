#!/usr/bin/env bash
set -euo pipefail

IMAGE_AARCH64="${IMAGE_AARCH64:-u2204_c118_p201-aarch64}"
IMAGE_X86_64="${IMAGE_X86_64:-u2204_c118_p230-amd64}"
PLATFORM_AARCH64="${PLATFORM_AARCH64:-linux/arm64/v8}"
PLATFORM_X86_64="${PLATFORM_X86_64:-linux/amd64}"
NAME_PREFIX="${NAME_PREFIX:-hanelso-alexnet}"

HOST_WS="${HOST_WS:-$PWD}"
HOST_DATA="${HOST_DATA:-$HOME/data}"
SHM_SIZE="${SHM_SIZE:-8g}"
DISPLAY_VAR="${DISPLAY:-:0}"

arch="$(uname -m)"
case "$arch" in
  aarch64|arm64) IMAGE="$IMAGE_AARCH64"; PLATFORM="$PLATFORM_AARCH64" ;;
  x86_64|amd64)  IMAGE="$IMAGE_X86_64";  PLATFORM="$PLATFORM_X86_64"  ;;
  *) echo "Unsupported arch: $arch"; exit 1 ;;
esac

CONTAINER_NAME="${NAME_PREFIX}-${arch}"

# /dev/dri 있으면 연결
DRI_ARGS=()
[ -e /dev/dri ] && DRI_ARGS=(--device /dev/dri)

# X11 접근 허용(원하면 종료 후 xhost -local:root로 되돌리기)
command -v xhost >/dev/null 2>&1 && xhost +local:root >/dev/null 2>&1 || true

# --- 핵심: Jetson/L4T( /etc/nv_tegra_release 존재 )이면 --runtime=nvidia 로 실행 ---
if [ -f /etc/nv_tegra_release ]; then
  GPU_ARGS=( --runtime nvidia
             -e NVIDIA_VISIBLE_DEVICES=all
             -e NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility )
else
  GPU_ARGS=( --gpus all
             -e NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility )
fi

X11_ARGS=( -e DISPLAY="$DISPLAY_VAR" -v /tmp/.X11-unix:/tmp/.X11-unix:rw -e QT_X11_NO_MITSHM=1 )
MOUNT_ARGS=( -v "$HOST_WS":/workspace -v "$HOST_DATA":/workspace/DATA )

exec docker run --rm -it \
  --name "$CONTAINER_NAME" \
  --platform "$PLATFORM" \
  "${GPU_ARGS[@]}" \
  "${X11_ARGS[@]}" \
  "${MOUNT_ARGS[@]}" \
  "${DRI_ARGS[@]}" \
  --shm-size="$SHM_SIZE" \
  "$IMAGE"

