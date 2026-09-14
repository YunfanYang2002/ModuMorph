#!/usr/bin/env bash
set -Eeuo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$root"
if (( $# < 3 )); then
  echo 'Usage: bash scripts/run_modumorph_eval_server.sh CONDA_ENV GPU [--no-keep-open] <runner arguments>' >&2
  exit 2
fi
environment="$1"
gpu="$2"
shift 2
keep_open=1
if [[ "${1:-}" == --no-keep-open ]]; then
  keep_open=0
  shift
fi
mkdir -p ./tmp
export TMPDIR="$root/tmp" TMP="$root/tmp" TEMP="$root/tmp"
launch="./tmp/modumorph_launch_$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -- "$launch"
output=''
arguments=("$@")
for (( index=0; index < ${#arguments[@]}; index++ )); do
  if [[ "${arguments[index]}" == --output ]]; then
    output="${arguments[index+1]}"
  fi
done
finish() {
  rc=$?
  trap - EXIT
  set +e
  printf '%s\n' "$rc" > "$launch/exit_code.txt"
  python - "$launch" "$output" <<'PY'
import pathlib,sys,zipfile
root=pathlib.Path.cwd().resolve()
launch=pathlib.Path(sys.argv[1]).resolve()
result=(root/sys.argv[2]).resolve() if sys.argv[2] else None
archive=launch.with_suffix('.zip')
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as out:
    sources=[launch]
    if result is not None and (root/'tmp').resolve() in result.parents and result.is_dir():
        sources.append(result)
    for folder in sources:
        for path in folder.rglob('*'):
            if path.is_file(): out.write(path,path.relative_to(root))
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
echo '[ENVIRONMENT]'
command -v conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$environment"
python -c 'import sys,torch,numpy,yaml,gym,mujoco_py,lxml; print(sys.executable); print(torch.__version__)'
export CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1
echo '[EVALUATION]'
python -u tools/run_modumorph_frozen_eval.py "${arguments[@]}"
echo '[PACKAGING]'
