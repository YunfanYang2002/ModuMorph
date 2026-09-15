#!/usr/bin/env bash
set -Eeuo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$root"
if (( $# < 1 )); then
  echo 'Usage: bash scripts/run_modumorph_evaluator_behavior_regression_server.sh GPU [--no-keep-open] [runner arguments]' >&2
  exit 2
fi
gpu="$1"
shift
keep_open=1
arguments=()
for argument in "$@"; do
  if [[ "$argument" == --no-keep-open ]]; then keep_open=0; else arguments+=("$argument"); fi
done
rmamorph_root="$root/../rmamorph"
reference_root="$rmamorph_root/tmp/morphadapt_canonical_student_formal_table2_20260905T111437Z"
output="$root/tmp/modumorph_evaluator_behavior_regression"
has_rmamorph=0
has_reference=0
has_output=0
for argument in "${arguments[@]}"; do
  [[ "$argument" == --rmamorph-root || "$argument" == --rmamorph-root=* ]] && has_rmamorph=1
  [[ "$argument" == --reference-root || "$argument" == --reference-root=* ]] && has_reference=1
  [[ "$argument" == --output || "$argument" == --output=* ]] && has_output=1
done
(( has_rmamorph )) || arguments+=(--rmamorph-root "$rmamorph_root")
(( has_reference )) || arguments+=(--reference-root "$reference_root")
(( has_output )) || arguments+=(--output tmp/modumorph_evaluator_behavior_regression)
mkdir -p "$root/tmp"
export TMPDIR="$root/tmp" TMP="$root/tmp" TEMP="$root/tmp"
export CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
launch="$root/tmp/modumorph_evaluator_behavior_regression_launch_$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -- "$launch"
finish() {
  rc=$?
  trap - EXIT
  set +e
  printf '%s\n' "$rc" > "$launch/exit_code.txt"
  python - "$launch" "$output" <<'PY'
import pathlib,sys,zipfile
root=pathlib.Path.cwd().resolve()
launch=pathlib.Path(sys.argv[1]).resolve()
output=pathlib.Path(sys.argv[2]).resolve()
archive=launch.with_suffix('.zip')
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as bundle:
    for directory in (launch,output):
        if not directory.is_dir(): continue
        for path in directory.rglob('*'):
            if path.is_file() and path.suffix in ('.json','.jsonl','.txt','.log'):
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
echo '[ENVIRONMENT]'
command -v python
python -c 'import sys,torch,numpy,yaml,gym,mujoco_py,lxml; print("PYTHON="+sys.executable); print("ENV_PREFIX="+sys.prefix); print("TORCH="+torch.__version__)'
echo '[PROVENANCE]'
hostname
git branch --show-current
git rev-parse HEAD
git status --short --untracked-files=no
git -C "$rmamorph_root" rev-parse HEAD
git -C "$rmamorph_root" status --short --untracked-files=no
echo "GPU=$gpu REFERENCE_ROOT=$reference_root"
echo '[ONE-EPISODE BEHAVIOR REGRESSION]'
python -u tools/run_modumorph_evaluator_behavior_regression.py "${arguments[@]}"
echo '[DONE: no ModuMorph training or formal Strict-OOD97 evaluation launched]'
