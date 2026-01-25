#!/bin/bash
# Run on the Linux box to install the systemd service.
# Usage: sudo ./install-service.sh [ /path/to/ingest ]
# Or:    sudo ./install-service.sh
#        then: sudo nano /etc/default/ragdoll-ingest  # set RAGDOLL_INGEST_PATH

set -e
BIN="$(cd -- "$(dirname "$0")" && pwd)"
INGEST="${1:-}"

cp -f "$BIN/ragdoll-ingest.service" /etc/systemd/system/
if [[ ! -f /etc/default/ragdoll-ingest ]]; then
  cp -f "$BIN/etc-default-ragdoll-ingest.example" /etc/default/ragdoll-ingest
  echo "Created /etc/default/ragdoll-ingest — set RAGDOLL_INGEST_PATH"
fi
if [[ -n "$INGEST" ]]; then
  # only override the path line if it exists
  if grep -q '^RAGDOLL_INGEST_PATH=' /etc/default/ragdoll-ingest; then
    sed -i "s|^RAGDOLL_INGEST_PATH=.*|RAGDOLL_INGEST_PATH=$INGEST|" /etc/default/ragdoll-ingest
  else
    echo "RAGDOLL_INGEST_PATH=$INGEST" >> /etc/default/ragdoll-ingest
  fi
  echo "Set RAGDOLL_INGEST_PATH=$INGEST"
fi

echo "To use a venv or custom path, run: sudo systemctl edit ragdoll-ingest"
echo "  [Service]"
echo "  WorkingDirectory=$BIN"
echo "  ExecStart=$BIN/.venv/bin/python -m ragdoll_ingest"
echo ""
echo "Then: sudo systemctl daemon-reload && sudo systemctl enable --now ragdoll-ingest"
