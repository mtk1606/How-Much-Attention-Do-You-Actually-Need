PY ?= .venv/bin/python
ART ?= artifacts

.PHONY: setup test lint smoke train-small eval-small figure-main match-report compute-summary

setup:
	uv venv .venv --python 3.11
	uv pip install --python $(PY) -e ".[dev]"

lint:
	$(PY) -m ruff check src tests scripts
	$(PY) -m ruff format --check src tests scripts

test:
	$(PY) -m pytest

# End-to-end on a laptop CPU in about a minute: train, evaluate, register, summarise. Not a result.
smoke:
	ATTNRATIO_ARTIFACTS=$(ART)/smoke $(PY) -m attnratio.cli sweep configs/sweeps/smoke.yaml --workers 1
	ATTNRATIO_ARTIFACTS=$(ART)/smoke $(PY) -m attnratio.cli analyze smoke --out-dir $(ART)/smoke/figures

# The CPU pilot sweep (about 1.5 h on 4 cores).
train-small:
	$(PY) -m attnratio.cli sweep configs/sweeps/pilot_mqar.yaml --workers 2 --threads 2

eval-small:
	$(PY) -m attnratio.cli analyze pilot_mqar --out-dir figures

figure-main:
	$(PY) -m attnratio.cli analyze pilot_mqar --out-dir figures

match-report:
	$(PY) -m attnratio.cli match-report configs/sweeps/*.yaml

compute-summary:
	$(PY) -m attnratio.cli compute-summary
