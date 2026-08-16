#!/usr/bin/env bash
# scripts/render_start.sh
#
# Render's disk is scratch-only, so rclone's config (the Filen remote
# credentials) has to be recreated on every boot rather than committed to
# the repo. Preferred path: upload rclone.conf as a Render "Secret File"
# named rclone.conf — Render places it at /etc/secrets/rclone.conf at
# runtime, and we copy it into place below. Falls back to RCLONE_CONF_B64
# (base64 of rclone.conf, set as a regular env var) for environments that
# don't support Secret Files.
set -euo pipefail

export PATH="$PWD/bin:$PATH"

if [ -f /etc/secrets/rclone.conf ]; then
  mkdir -p "$HOME/.config/rclone"
  cp /etc/secrets/rclone.conf "$HOME/.config/rclone/rclone.conf"
elif [ -n "${RCLONE_CONF_B64:-}" ]; then
  mkdir -p "$HOME/.config/rclone"
  echo "$RCLONE_CONF_B64" | base64 -d > "$HOME/.config/rclone/rclone.conf"
fi

exec python server.py
