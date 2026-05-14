#!/usr/bin/env bash
# Stop RAGDoll systemd units, pull latest code, reinstall package, reload units, start enabled services.
# Run from the repo root (same directory as this script). Requires: git, sudo for systemctl, pip in .venv or pip3.
#
# Optional environment:
#   RAGDOLL_GIT_REMOTE   default: origin
#   RAGDOLL_GIT_BRANCH   default: main
#   RAGDOLL_SERVICES     space-separated unit basenames (default: ragdoll-mcp ragdoll-web ragdoll-api ragdoll-ingest)
#                        Order is stop (first to last), then start (last to first) so ingest starts last.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

REMOTE="${RAGDOLL_GIT_REMOTE:-origin}"
BRANCH="${RAGDOLL_GIT_BRANCH:-main}"
# Stop: dependents first. Start: reverse (ingest before or after api — ingest does not depend on api).
DEFAULT_SERVICES=(ragdoll-mcp ragdoll-web ragdoll-api ragdoll-ingest)
if [[ -n "${RAGDOLL_SERVICES:-}" ]]; then
  # shellcheck disable=2206
  SERVICES=($RAGDOLL_SERVICES)
else
  SERVICES=("${DEFAULT_SERVICES[@]}")
fi

die() { echo "error: $*" >&2; exit 1; }

command -v git >/dev/null || die "git not found"
command -v sudo >/dev/null || die "sudo not found"

stop_one() {
  local s=$1
  if systemctl list-unit-files --no-pager "${s}.service" 2>/dev/null | grep -qF "${s}.service"; then
    sudo systemctl stop "$s" 2>/dev/null || true
  fi
}

start_one_if_enabled() {
  local s=$1
  if systemctl list-unit-files --no-pager "${s}.service" 2>/dev/null | grep -qF "${s}.service"; then
    if systemctl is-enabled --quiet "$s" 2>/dev/null; then
      echo "starting (enabled): $s"
      sudo systemctl start "$s" || echo "warning: systemctl start $s failed (see journalctl -u $s)" >&2
    else
      echo "skip start (not enabled): $s"
    fi
  else
    echo "skip (no unit file): $s"
  fi
}

echo "==> stopping services (if installed)..."
for s in "${SERVICES[@]}"; do
  stop_one "$s"
done

echo "==> git pull $REMOTE $BRANCH"
git pull "$REMOTE" "$BRANCH"

echo "==> pip install -e ."
if [[ -x "$REPO_ROOT/.venv/bin/pip" ]]; then
  "$REPO_ROOT/.venv/bin/pip" install -e "$REPO_ROOT"
elif command -v pip3 >/dev/null; then
  pip3 install -e "$REPO_ROOT"
else
  die "no .venv/bin/pip or pip3; create a venv or install pip"
fi

echo "==> systemctl daemon-reload"
sudo systemctl daemon-reload

echo "==> starting enabled services..."
for (( idx=${#SERVICES[@]}-1 ; idx>=0 ; idx-- )); do
  start_one_if_enabled "${SERVICES[idx]}"
done

echo "==> done. status:"
for s in "${SERVICES[@]}"; do
  if systemctl list-unit-files --no-pager "${s}.service" 2>/dev/null | grep -qF "${s}.service"; then
    st="$(systemctl is-active "$s" 2>/dev/null || echo unknown)"
    printf '  %-18s %s\n' "$s" "$st"
  fi
done
