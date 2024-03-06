SHELL := /bin/bash

build:
	python3 -m venv ./ ; \
	source ./bin/activate ; \
	pip3 install -r requirements.txt ; \
	python3 downloads.py ; \
