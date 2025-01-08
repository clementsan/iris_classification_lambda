.PHONY: lint
lint:  # Lints our code.
	python -m pylint ./

.PHONY: black
black:  # Formats our code.
	python -m black ./

.PHONY: test
test:  # Runs tests.
	python -m pytest tests


