#!/usr/bin/env bash
set -euo pipefail

# Volúmenes nuevos llegan como root
sudo chown -R "$(id -u):$(id -g)" ~/.claude ~/.cache

# SSH (Azure DevOps)
mkdir -p ~/.ssh && chmod 700 ~/.ssh
cp -r ~/.ssh-host/. ~/.ssh/ 2>/dev/null || true
chmod 600 ~/.ssh/* 2>/dev/null || true
chmod 644 ~/.ssh/*.pub 2>/dev/null || true
ssh-keyscan -H ssh.dev.azure.com >> ~/.ssh/known_hosts 2>/dev/null || true

# Git
git config --global --add safe.directory /llm-studies

# Python
if [ -f pyproject.toml ]; then
    uv sync
    uv run python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.version.cuda)" || true
fi
