#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# Run from a host venv with requirements-tools.txt installed. Never overwrites .env.
[[ -f .env ]] || python manage.py init-env
python manage.py doctor
# --quiet avoids printing the interpolated credentials.
docker compose config --quiet
docker compose build
docker compose pull vllm qdrant postgres edge
python manage.py doctor --containers
python manage.py record-images
docker compose up -d --wait --wait-timeout 3600
python manage.py smoke
printf '\nLive smoke checks passed. Evidence is in deployment/artifacts/.\n'
printf 'Next, explicitly index your corpus: python manage.py seed-recipes --index recipes-v1\n'
