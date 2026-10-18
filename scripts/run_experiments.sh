#!/bin/sh
# Phase 5 experiment queue (RQ1-RQ3). Runs every job sequentially; a failing job is
# logged and the queue continues. Progress: logs/queue_status.txt, one log per job.
#
# Usage: sh scripts/run_experiments.sh [PID_TO_WAIT_FOR]
#   If a PID is given, the queue starts after that process exits.

cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
STATUS=logs/queue_status.txt
mkdir -p logs

if [ -n "$1" ]; then
    echo "$(date '+%F %T') waiting for PID $1" >> "$STATUS"
    while kill -0 "$1" 2>/dev/null; do sleep 30; done
fi

run() {
    name=$1; shift
    if [ -f "logs/$name.done" ]; then
        echo "$(date '+%F %T') SKIP  $name (already done)" >> "$STATUS"
        return
    fi
    echo "$(date '+%F %T') START $name" >> "$STATUS"
    if "$@" > "logs/$name.log" 2>&1; then
        touch "logs/$name.done"
        echo "$(date '+%F %T') OK    $name" >> "$STATUS"
    else
        echo "$(date '+%F %T') FAIL  $name (see logs/$name.log)" >> "$STATUS"
    fi
}

# ---- RQ2: label efficiency (seed 42; the 100% runs come from RQ1) ----
for lf in 0.01 0.05 0.1; do
    run "rq2_rf_lf$lf"        $PY scripts/train_sklearn.py --model rf --label-fraction "$lf"
    run "rq2_cnn_lf$lf"       $PY scripts/train_torch.py --model cnn --label-fraction "$lf"
    run "rq2_tf_pt_lf$lf"     $PY scripts/finetune.py --label-fraction "$lf"
    run "rq2_tf_scratch_lf$lf" $PY scripts/finetune.py --from-scratch --label-fraction "$lf"
done
run "rq2_tf_scratch_lf1"      $PY scripts/finetune.py --from-scratch

# ---- RQ3: IP/port masking ablation (seed 42; 'ip' is the default from RQ1) ----
for m in none ip_port; do
    run "rq3_cnn_$m"          $PY scripts/train_torch.py --model cnn --mask "$m"
    run "rq3_pretrain_$m"     $PY scripts/pretrain.py --mask "$m"
    run "rq3_tf_pt_$m"        $PY scripts/finetune.py --mask "$m"
done

# ---- RQ1: extra seeds (Transformer fine-tunes reuse the seed-42 pre-trained encoder) ----
for s in 43 44; do
    run "rq1_rf_seed$s"       $PY scripts/train_sklearn.py --model rf --seed "$s"
    run "rq1_cnn_seed$s"      $PY scripts/train_torch.py --model cnn --seed "$s"
    run "rq1_gru_seed$s"      $PY scripts/train_torch.py --model gru --seed "$s"
    run "rq1_tf_pt_seed$s"    $PY scripts/finetune.py --seed "$s" --pretrain-seed 42
done

echo "$(date '+%F %T') QUEUE FINISHED" >> "$STATUS"
