# Pipeline entry points. Run in order:
#   make setup -> make download -> make preprocess -> make rf -> make cnn -> make pretrain -> make finetune
#   -> make label-efficiency -> make ablation -> make report
# Extra CLI overrides can be passed via ARGS, e.g.:
#   make preprocess ARGS="--max-sessions-per-class 200"

PYTHON_BIN ?= python3.11
VENV       := .venv
PY         := $(VENV)/bin/python
ARGS       ?=

.PHONY: setup setup-xgb download preprocess rf xgb cnn gru pretrain finetune experiments label-efficiency ablation report test clean-processed

setup:
	$(PYTHON_BIN) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e . --no-deps

# Optional: XGBoost baseline (macOS needs `brew install libomp` first)
setup-xgb:
	$(PY) -m pip install "xgboost>=2.0"

download:
	$(PY) scripts/download_ustc.py

preprocess:
	$(PY) scripts/preprocess.py $(ARGS)

rf:
	$(PY) scripts/train_sklearn.py --model rf $(ARGS)

xgb:
	$(PY) scripts/train_sklearn.py --model xgb $(ARGS)

cnn:
	$(PY) scripts/train_torch.py --model cnn $(ARGS)

gru:
	$(PY) scripts/train_torch.py --model gru $(ARGS)

pretrain:
	$(PY) scripts/pretrain.py $(ARGS)

finetune:
	$(PY) scripts/finetune.py $(ARGS)

# Full Phase 5 queue (RQ1 extra seeds, RQ2, RQ3); safe to re-run, finished jobs are skipped
experiments:
	sh scripts/run_experiments.sh

label-efficiency: experiments


ablation: experiments


report:
	$(PY) scripts/report.py $(ARGS)

test:
	$(PY) -m pytest

clean-processed:
	rm -rf data/processed/*
