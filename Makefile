DATA ?= data/processed/cube.nc

.PHONY: help install test test-all smoke train experiments clean

help:
	@echo "OceanEmbed - available commands"
	@echo "  make install      install Python dependencies"
	@echo "  make test         fast unit tests (~5 s)"
	@echo "  make test-all     all tests incl. end-to-end (~2 min)"
	@echo "  make smoke        end-to-end run on a synthetic cube"
	@echo "  make train        train the main model, 3 seeds (DATA=path/to/cube.nc)"
	@echo "  make experiments  full study: 4 models x 3 seeds + evaluation (DATA=...)"
	@echo "  make clean        remove caches and run artifacts"

install:
	pip install -r requirements.txt

test:
	python -m pytest -m "not slow"

test-all:
	python -m pytest

smoke:
	python scripts/smoke_test.py

train:
	for s in 0 1 2; do python -m oceanembed train --config configs/main.yaml --data $(DATA) --out artifacts/runs/oceanembed_cyclone_s$$s --seed $$s; done

experiments:
	bash scripts/run_experiments.sh $(DATA)

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache artifacts
