.PHONY: init install dev test version clean

init:
	pip install -r requirements.txt

install:
	pip install .

dev:
	pip install -e .

# run_checks.py is a standalone consistency suite, not a pytest suite:
# it rebuilds the coefficients, power spectra, boxes and the burstiness
# numbers and checks them against the published tables. ~2 minutes.
test:
	python run_checks.py

version:
	@python -c "g={'__file__':'oLIMpus/_version.py'}; exec(open('oLIMpus/_version.py').read(), g); print(g['get_version']())"

clean:
	rm -rf build dist *.egg-info
	find . -name __pycache__ -type d -exec rm -rf {} +
