#!/usr/bin/env bash
# scripts/render_build.sh
#
# Render's Python runtime doesn't ship rclone, and storage/filen.py and
# config/backup.py both shell out to it. This installs a static rclone
# binary into ./bin (no root/apt needed) as part of Render's build step.
set -euo pipefail

pip install -r requirements.txt

if [ ! -f ./bin/rclone ]; then
  mkdir -p ./bin
  curl -fsSL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o /tmp/rclone.zip
  unzip -o /tmp/rclone.zip -d /tmp/rclone-extract
  cp /tmp/rclone-extract/rclone-*-linux-amd64/rclone ./bin/rclone
  chmod +x ./bin/rclone
  rm -rf /tmp/rclone.zip /tmp/rclone-extract
fi
