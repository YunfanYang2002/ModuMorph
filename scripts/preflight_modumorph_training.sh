#!/usr/bin/env bash
set -Eeuo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$root/scripts/run_modumorph_training_server.sh" "${1:?provide one physical GPU ID}" preflight "${@:2}"
