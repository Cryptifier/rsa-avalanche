.PHONY: all build test deploy install-deps web

all: build

build:
	cargo build

test:
	cargo test

deploy:
	./scripts/make_deploy.sh

demo:
	RUNS=1 SEED_START=2100000 AVALANCHE_BATCHES=50 ./scripts/run_small_public_key_demo.sh

install-deps:
	./scripts/install_deps.sh

web:
	cargo build --target wasm32-unknown-unknown --bin viewer
	wasm-bindgen --target web --out-dir web --out-name viewer target/wasm32-unknown-unknown/debug/viewer.wasm
	cargo run --bin server -- --log-dir logs --web-dir web --addr 127.0.0.1:8080

