PYTHON ?= python3
VENV ?= .venv
SLAM_PYTHONPATH := src$(if $(PYTHONPATH),:$(PYTHONPATH))
ifeq ($(wildcard $(VENV)/bin/python),)
RUN := $(PYTHON)
PIP := $(PYTHON) -m pip
else
RUN := $(VENV)/bin/python
PIP := $(RUN) -m pip
endif

.PHONY: venv install-pi install-laptop check record reconstruct regression regression-score

venv:
	$(PYTHON) -m venv $(VENV)

install-pi: venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[pi]"

install-laptop: venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[laptop]"

check:
	find src/slam -name '*.py' -not -path '*/__pycache__/*' -print0 | xargs -0 $(PYTHON) -m py_compile

record:
	PYTHONPATH=$(SLAM_PYTHONPATH) $(RUN) -m slam record

reconstruct:
	PYTHONPATH=$(SLAM_PYTHONPATH) $(RUN) -m slam reconstruct

regression:
	PYTHONPATH=$(SLAM_PYTHONPATH) $(RUN) -m slam regression

regression-score:
	PYTHONPATH=$(SLAM_PYTHONPATH) $(RUN) -m slam regression-score
