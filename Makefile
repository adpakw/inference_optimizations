PYTHON := .venv/bin/python

.PHONY: benchmark
benchmark:
	$(PYTHON) -m benchmark

.PHONY: run_baseline
run_baseline:
	$(PYTHON) -m baseline.app


