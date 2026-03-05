.PHONY: all build test deploy

all: build

build:
	cargo build

test:
	cargo test

deploy:
	./scripts/make_deploy.sh
