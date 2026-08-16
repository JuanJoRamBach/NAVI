#!/usr/bin/env bash
# scripts/render_start.sh
#
# Render's disk is scratch-only, so rclone's config (the Filen remote
# credentials) has to be recreated on every boot from an env var rather
# than committed to the repo. Set RCLONE_CONF_B64 in Render's dashboard
# to the base64 of your local rclone.conf (the one already configured
# and tested against the "filen" remote):
#
#   base64 -w0 ~/.config/rclone/rclone.conf   (Linux/Mac)
#   [Convert]::ToBase64String([IO.File]::ReadAllBytes("$env:APPDATA\rclone\rclone.conf"))   (Windows PowerShell)
#
# rclone.conf can contain an obscured Filen password — treat RCLONE_CONF_B64
# as a secret in Render's env var settings, same as the API keys.
set -euo pipefail

export PATH="$PWD/bin:$PATH"

if [ -n "${RCLONE_CONF_B64:-}" ]; then
  mkdir -p "$HOME/.config/rclone"
  echo "$RCLONE_CONF_B64" | base64 -d > "$HOME/.config/rclone/rclone.conf"
fi

exec python server.py
