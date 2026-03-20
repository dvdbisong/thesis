# Learning Automata Kelp Detection Prototype
# Makefile for experiment management

.PHONY: run run-name list clean clean-all setup test help \
	download-multitemporal-dry download-multitemporal \
	check-multitemporal-status download-multitemporal-gcs \
	preprocess-multitemporal validate-multitemporal multitemporal-all

# Default config file
CONFIG ?= config.yaml

# Conda environment and Python interpreter
CONDA_ENV = uvic
CONDA_ACTIVATE = eval "$$(~/anaconda3/bin/conda shell.zsh hook)" && conda activate $(CONDA_ENV)
PYTHON = python3

# Directories
CODE_DIR = code
EXP_DIR = experiments
RESULTS_DIR = results

#-------------------------------------------------------------------------------
# Main targets
#-------------------------------------------------------------------------------

## Run experiment with default config
run:
	$(CONDA_ACTIVATE) && PYTHONPATH=. $(PYTHON) $(CODE_DIR)/experiments/run_experiment.py --config $(CONFIG)

## Run experiment with custom name: make run-name NAME=my_experiment
run-name:
	$(CONDA_ACTIVATE) && PYTHONPATH=. $(PYTHON) $(CODE_DIR)/experiments/run_experiment.py --config $(CONFIG) --name $(NAME)

## Run experiment with custom config: make run-config CFG=my_config.yaml
run-config:
	$(CONDA_ACTIVATE) && PYTHONPATH=. $(PYTHON) $(CODE_DIR)/experiments/run_experiment.py --config $(CFG)

#-------------------------------------------------------------------------------
# Experiment management
#-------------------------------------------------------------------------------

## List all experiments
list:
	@if [ -f $(EXP_DIR)/index.json ]; then \
		echo "=== Experiment Index ==="; \
		$(PYTHON) -m json.tool $(EXP_DIR)/index.json; \
	else \
		echo "No experiments found. Run 'make run' to create one."; \
	fi

## Show last experiment results
last:
	@if [ -f $(EXP_DIR)/index.json ]; then \
		LAST=$$($(PYTHON) -c "import json; d=json.load(open('$(EXP_DIR)/index.json')); print(d[-1]['path'] if d else '')"); \
		if [ -n "$$LAST" ]; then \
			echo "=== Last Experiment: $$LAST ==="; \
			cat $$LAST/metrics.json | $(PYTHON) -m json.tool; \
		fi \
	else \
		echo "No experiments found."; \
	fi

#-------------------------------------------------------------------------------
# Data preparation
#-------------------------------------------------------------------------------

## Create train/val/test splits
splits:
	$(CONDA_ACTIVATE) && PYTHONPATH=. $(PYTHON) $(CODE_DIR)/preprocessing/create_splits.py --config $(CONFIG)

## Verify data integrity
verify-data:
	$(CONDA_ACTIVATE) && PYTHONPATH=. $(PYTHON) $(CODE_DIR)/preprocessing/data_loader.py --verify

#-------------------------------------------------------------------------------
# Multi-Temporal Data (Phase 0.5)
#-------------------------------------------------------------------------------

## Download multi-temporal Sentinel-2 imagery via GEE (dry run)
download-multitemporal-dry:
	$(CONDA_ACTIVATE) && PYTHONPATH=. $(PYTHON) -m src.preprocessing.gee_download \
		--output data/bc_sentinel2_multitemporal/raw/ \
		--dry-run

## Export multi-temporal Sentinel-2 imagery to GCS via GEE
download-multitemporal:
	$(CONDA_ACTIVATE) && PYTHONPATH=. $(PYTHON) -m src.preprocessing.gee_download \
		--output data/bc_sentinel2_multitemporal/raw/

## Check GCS export status
check-multitemporal-status:
	$(CONDA_ACTIVATE) && PYTHONPATH=. $(PYTHON) -m src.preprocessing.gee_download --check-status

## Download multi-temporal imagery from GCS to local
download-multitemporal-gcs:
	$(CONDA_ACTIVATE) && PYTHONPATH=. $(PYTHON) -m src.preprocessing.gee_download \
		--download-from-gcs \
		--output data/bc_sentinel2_multitemporal/raw/

## Preprocess multi-temporal imagery to tiles
preprocess-multitemporal:
	$(CONDA_ACTIVATE) && PYTHONPATH=. $(PYTHON) -m src.preprocessing.preprocess_multitemporal \
		--input data/bc_sentinel2_multitemporal/raw/ \
		--output data/bc_sentinel2_multitemporal/Tiles/ \
		--auxiliary "data/bc_sentinel2/new/Masks 10 scenes/"

## Validate multi-temporal tiles
validate-multitemporal:
	@echo "Validating tiles in data/bc_sentinel2_multitemporal/Tiles/"
	@for dir in data/bc_sentinel2_multitemporal/Tiles/*/; do \
		echo "Validating $$dir"; \
		$(CONDA_ACTIVATE) && PYTHONPATH=. $(PYTHON) -m src.preprocessing.tile_creator validate --dir "$$dir"; \
	done

## Full multi-temporal pipeline
multitemporal-all: download-multitemporal preprocess-multitemporal validate-multitemporal

#-------------------------------------------------------------------------------
# Baselines
#-------------------------------------------------------------------------------

## Run all baselines (random, fixed, oracle)
baselines:
	$(CONDA_ACTIVATE) && PYTHONPATH=. $(PYTHON) $(CODE_DIR)/experiments/baselines.py --config $(CONFIG)

#-------------------------------------------------------------------------------
# Multi-seed and Batch Experiments
#-------------------------------------------------------------------------------

# Number of seeds (default: 5)
N_SEEDS ?= 5

## Run experiment with multiple seeds: make run-seeds
run-seeds:
	$(CONDA_ACTIVATE) && PYTHONPATH=. $(PYTHON) -m code.experiments.experiment_runner \
		--config $(CONFIG) --n-seeds $(N_SEEDS)

## Run experiment with specific seeds: make run-seeds-custom SEEDS="42 123 456"
run-seeds-custom:
	$(CONDA_ACTIVATE) && PYTHONPATH=. $(PYTHON) -m code.experiments.experiment_runner \
		--config $(CONFIG) --seeds $(SEEDS)

## Run LR-I experiment with multiple seeds
run-lri:
	$(CONDA_ACTIVATE) && PYTHONPATH=. $(PYTHON) -m code.experiments.experiment_runner \
		--config configs/algorithms/lri.yaml --n-seeds $(N_SEEDS) --name lri_baseline

## Run LR-P experiment with multiple seeds
run-lrp:
	$(CONDA_ACTIVATE) && PYTHONPATH=. $(PYTHON) -m code.experiments.experiment_runner \
		--config configs/algorithms/lrp.yaml --n-seeds $(N_SEEDS) --name lrp_baseline

## Run VSLA experiment with multiple seeds
run-vsla:
	$(CONDA_ACTIVATE) && PYTHONPATH=. $(PYTHON) -m code.experiments.experiment_runner \
		--config configs/algorithms/vsla.yaml --n-seeds $(N_SEEDS) --name vsla_adaptive

## Run Pursuit experiment with multiple seeds
run-pursuit:
	$(CONDA_ACTIVATE) && PYTHONPATH=. $(PYTHON) -m code.experiments.experiment_runner \
		--config configs/algorithms/pursuit.yaml --n-seeds $(N_SEEDS) --name pursuit_baseline

## Run Estimator experiment with multiple seeds
run-estimator:
	$(CONDA_ACTIVATE) && PYTHONPATH=. $(PYTHON) -m code.experiments.experiment_runner \
		--config configs/algorithms/estimator.yaml --n-seeds $(N_SEEDS) --name estimator_seri

#-------------------------------------------------------------------------------
# Statistical Analysis
#-------------------------------------------------------------------------------

## Run statistical analysis on all results
stats:
	$(CONDA_ACTIVATE) && PYTHONPATH=. $(PYTHON) -m code.experiments.statistical_analysis \
		--results-dir $(RESULTS_DIR) --latex

## Compare specific experiments: make compare EXPS="lri_baseline lrp_baseline"
compare:
	$(CONDA_ACTIVATE) && PYTHONPATH=. $(PYTHON) -m code.experiments.statistical_analysis \
		--results-dir $(RESULTS_DIR) --experiments $(EXPS) --latex

#-------------------------------------------------------------------------------
# Cleanup
#-------------------------------------------------------------------------------

## Clean experiment outputs (keeps index)
clean:
	rm -rf $(EXP_DIR)/exp_*
	rm -rf $(RESULTS_DIR)/*

## Clean everything including index
clean-all: clean
	rm -f $(EXP_DIR)/index.json

#-------------------------------------------------------------------------------
# Setup
#-------------------------------------------------------------------------------

## Install dependencies in kelp conda environment
setup:
	$(CONDA_ACTIVATE) && pip install -r requirements.txt

## Run tests
test:
	$(CONDA_ACTIVATE) && PYTHONPATH=. $(PYTHON) -m pytest tests/ -v

#-------------------------------------------------------------------------------
# Help
#-------------------------------------------------------------------------------

## Show this help message
help:
	@echo "Learning Automata Kelp Detection Prototype"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^## ' Makefile | sed 's/## /  /'
	@echo ""
	@echo "Examples:"
	@echo "  make run                    Run experiment with config.yaml"
	@echo "  make run-name NAME=test1    Run experiment with custom name"
	@echo "  make run-seeds              Run experiment with 5 seeds"
	@echo "  make run-lri                Run LR-I with multiple seeds"
	@echo "  make run-lrp                Run LR-P with multiple seeds"
	@echo "  make run-vsla               Run VSLA with multiple seeds"
	@echo "  make run-pursuit            Run Pursuit with multiple seeds"
	@echo "  make stats                  Run statistical analysis"
	@echo "  make compare EXPS='a b'     Compare specific experiments"
	@echo "  make baselines              Run baseline comparisons"
	@echo "  make list                   List all experiments"
	@echo ""
	@echo "Multi-Temporal Data (Phase 0.5):"
	@echo "  make download-multitemporal-dry   Query GEE without exporting"
	@echo "  make download-multitemporal       Export to GCS via GEE"
	@echo "  make check-multitemporal-status   Check GCS export status"
	@echo "  make download-multitemporal-gcs   Download from GCS to local"
	@echo "  make preprocess-multitemporal     Process to tiles"
	@echo "  make validate-multitemporal       Validate tile integrity"
	@echo "  make multitemporal-all            Full pipeline"

# Default target
.DEFAULT_GOAL := help
