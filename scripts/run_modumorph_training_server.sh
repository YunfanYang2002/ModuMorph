#!/usr/bin/env bash
set -Eeuo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$root"
if (( $# < 2 )); then
  echo 'Usage: bash scripts/run_modumorph_training_server.sh GPU preflight|benchmark|pilot [--resume] [--no-keep-open]' >&2
  exit 2
fi
gpu="$1"
mode="$2"
shift 2
keep_open=1
arguments=()
for argument in "$@"; do
  if [[ "$argument" == --no-keep-open ]]; then keep_open=0; else arguments+=("$argument"); fi
done
mkdir -p ./tmp
export TMPDIR="$root/tmp" TMP="$root/tmp" TEMP="$root/tmp"
export MPLCONFIGDIR="$root/tmp/matplotlib" XDG_CACHE_HOME="$root/tmp/cache"
export CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
launch="./tmp/modumorph_training_launch_$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -- "$launch"
finish() {
  rc=$?
  trap - EXIT
  set +e
  printf '%s\n' "$rc" > "$launch/exit_code.txt"
  python - "$launch" "$mode" <<'PY'
import pathlib,sys,zipfile
root=pathlib.Path.cwd().resolve()
launch=root/sys.argv[1]
archive=launch.with_suffix('.zip')
outputs={'benchmark':root/'tmp/modumorph_throughput_s1409','pilot':root/'tmp/modumorph_pilot_s1409_10m'}
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as bundle:
    for directory in [launch,outputs.get(sys.argv[2])]:
        if directory is None or not directory.is_dir(): continue
        for path in directory.rglob('*'):
            # Compact diagnostics; model/resume binaries remain in original output directory.
            if path.is_file() and path.suffix in ('.json','.jsonl','.yaml','.log','.txt'):
                bundle.write(path,path.relative_to(root))
print('OUTPUT_ZIP='+str(archive))
PY
  package_rc=$?
  if (( package_rc != 0 )); then
    echo "PACKAGING_FAILED=$package_rc; evidence remains in $launch" >&2
    if (( rc == 0 )); then rc="$package_rc"; fi
  fi
  echo "RUN_EXIT_CODE=$rc"
  if (( keep_open )) && [[ -t 0 ]]; then
    echo 'Terminal retained. Type exit to close.'
    exec bash -i
  fi
  exit "$rc"
}
trap finish EXIT
exec > >(tee -a "$launch/console.log") 2>&1
echo '[ENVIRONMENT: verify the currently activated training environment]'
command -v python
python -c 'import sys,torch,numpy,yaml,gym,mujoco_py,lxml,cloudpickle,psutil,tensorboard; print("PYTHON="+sys.executable); print("ENV_PREFIX="+sys.prefix); print("TORCH="+torch.__version__)'
echo '[PROVENANCE]'
hostname
git branch --show-current
git rev-parse HEAD
git status --short --untracked-files=no
echo "GPU=$gpu MODE=$mode"
echo '[PREFLIGHT / TRAINING]'
python -u tools/run_modumorph_training.py "$mode" "${arguments[@]}"
echo '[DONE: no evaluation launched]'
