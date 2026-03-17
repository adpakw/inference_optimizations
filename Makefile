PYTHON := .venv/bin/python

.PHONY: run_baseline
run_baseline:
	$(PYTHON) baseline/app.py

.PHONY: run_onnx
run_onnx:
	$(PYTHON) onnx/app.py

.PHONY: convert_onnx
convert_onnx:
	$(PYTHON) onnx/convert_model.py

.PHONY: benchmark
benchmark:
	$(PYTHON) benchmark.py