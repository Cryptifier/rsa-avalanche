.PHONY: all build test deploy install-deps

all: build

build:
	cargo build

test:
	cargo test

deploy:
	./scripts/make_deploy.sh

install-deps:
	./scripts/install_deps.sh
