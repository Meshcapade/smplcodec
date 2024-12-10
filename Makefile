# Convenience setting to avoid needing tab characters in recipes
.RECIPEPREFIX = >

.PHONY: default
default: help

# generate help info from comments: thanks to https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
.PHONY: help
help: ## help information about make commands
> @grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: build
build: ## build the project in a local virtual environment
> poetry build

.PHONY: install
install: ## install the project in a local virtual environment
> poetry install

.PHONY: test
test: ## run unit tests
> poetry run pytest

.PHONY: check
check: ## run typechecker, linter and formatter in "report mode"
> poetry run pyright
> poetry run ruff check
> poetry run ruff format --check

.PHONY: fix
fix: ## run linter and formatter in "fix mode"
> poetry run ruff check --fix
> poetry run ruff format
