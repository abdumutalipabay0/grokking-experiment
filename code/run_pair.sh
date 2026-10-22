set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
mkdir -p runs
"$PY" code/train.py --validate > validation.json
for seed in 0 1 2; do
  for wd in 1 0; do
    "$PY" code/train.py --seed "$seed" --wd "$wd" --steps 20000 --output "runs/wd${wd}-seed${seed}" > "runs/wd${wd}-seed${seed}.log" 2>&1
  done
done
date -u +%FT%TZ > COMPLETE
