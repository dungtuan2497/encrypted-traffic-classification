# Pipeline entry points. Run in order:
#   make setup -> make download -> make preprocess -> make rf -> make cnn -> make pretrain -> make finetune
#   -> make label-efficiency -> make ablation -> make report
# Extra CLI overrides can be passed via ARGS, e.g.:
#   make preprocess ARGS="--max-sessions-per-class 200"

PYTHON_BIN ?= python3.11
VENV       := .venv
PY         := $(VENV)/bin/python
ARGS       ?=

.PHONY: setup setup-xgb download preprocess rf xgb cnn gru pretrain finetune label-efficiency ablation report test clean-processed

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
	@echo "[Phase 3] not implemented yet"

gru:
	@echo "[Phase 3, optional] not implemented yet"

pretrain:
	@echo "[Phase 4] not implemented yet"

finetune:
	@echo "[Phase 4] not implemented yet"

label-efficiency:
	@echo "[Phase 5] not implemented yet"

ablation:
	@echo "[Phase 5] not implemented yet"

report:
	@echo "[Phase 6] not implemented yet"

test:
	$(PY) -m pytest

clean-processed:
	rm -rf data/processed/*
