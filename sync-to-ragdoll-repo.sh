#!/bin/bash
# Sync this RAGDoll project to your "ragdoll" repo.
#
# 1. Set RAGDOLL_REPO_URL to your repo (e.g. https://github.com/you/ragdoll.git or git@github.com:you/ragdoll.git)
# 2. Run:  ./sync-to-ragdoll-repo.sh
#
# If the remote already has commits (e.g. README from GitHub), first run:
#   git pull origin main --allow-unrelated-histories
#   # resolve conflicts if any, then re-run this script or: git push -u origin main

set -e
cd "$(dirname "$0")"

if [[ -z "$RAGDOLL_REPO_URL" ]]; then
  echo "Set RAGDOLL_REPO_URL to your ragdoll repo (e.g. https://github.com/you/ragdoll.git)"
  exit 1
fi

if [[ ! -d .git ]]; then
  git init
  git add -A
  git commit -m "Initial commit: RAGDoll ingest service"
  git branch -M main
fi

if ! git remote get-url origin 2>/dev/null; then
  git remote add origin "$RAGDOLL_REPO_URL"
else
  git remote set-url origin "$RAGDOLL_REPO_URL"
fi

git add -A
git status
if [[ -n "$(git status --porcelain)" ]]; then
  git commit -m "Sync: RAGDoll ingest updates"
fi

git push -u origin main
