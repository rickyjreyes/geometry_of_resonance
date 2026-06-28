PYTHON ?= python3
ARTIFACT_DIR ?= reproducibility/artifacts

.PHONY: install reproduce verify clean

install:
	$(PYTHON) -m pip install --requirement reproducibility/requirements.lock

reproduce: clean
	$(PYTHON) reproducibility/finite_band_reference.py --output-dir $(ARTIFACT_DIR)

verify:
	$(PYTHON) reproducibility/verify_hashes.py --artifact-dir $(ARTIFACT_DIR)

clean:
	rm -rf $(ARTIFACT_DIR)
