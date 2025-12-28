# Makefile

# Specify the Python version
PYTHON_BIN := python3.11
LAMBDA_VENV := .lambda_venv
PACKAGE_NAME := lambda_package.zip

SITE_PACKAGES := $(shell $(LAMBDA_VENV)/bin/python -c "import site; print(site.getsitepackages()[0])")

# Default target executed when no arguments are given to make.
default: clean package

# Setup a virtual environment
venv:
	$(PYTHON_BIN) -m venv $(LAMBDA_VENV)
	$(LAMBDA_VENV)/bin/pip install -U pip

# Install dependencies into the virtual environment
dependencies: venv
	$(LAMBDA_VENV)/bin/pip install -r requirements.txt

# Package the virtual environment libraries and your lambda function into a zip
package: dependencies
# deps
	SITE_PACKAGES=`$(LAMBDA_VENV)/bin/python -c "import site; print(site.getsitepackages()[0])"` && \
	cd $$SITE_PACKAGES && zip -r9 $(CURDIR)/$(PACKAGE_NAME) .
# code (tutti i .py nella root dello zip)
	zip -g $(PACKAGE_NAME) *.py

# Clean up the environment
clean:
	rm -rf $(LAMBDA_VENV)
	rm -f $(PACKAGE_NAME)

.PHONY: default venv dependencies package clean
