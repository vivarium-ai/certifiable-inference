EXES :=
DIST :=

# Versioning
VERSION ?= $(shell (git describe --tags 2>/dev/null || echo "develop") | sed 's/^v//')
REVISION ?= $(shell git rev-parse --short HEAD)

export VERSION
export REVISION

# Build configuration (overrideable)
BUILD_DIR ?= build
BUILD_TYPE ?= Release
GENERATOR ?= Ninja

CMAKE ?= cmake
CTEST ?= ctest

# ccache (enabled by default if available)
CCACHE ?= ccache
CCACHE_DIR ?= $(HOME)/.ccache
export CCACHE_DIR

CMAKE_CACHE_ARGS := \
	-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
	-DCMAKE_C_COMPILER_LAUNCHER=$(CCACHE) \
	-DCMAKE_CXX_COMPILER_LAUNCHER=$(CCACHE)

.PHONY: all help deps config build test install release clean

all: test

##@ Dependencies
deps: ## Install project dependencies
	./scripts/deps.sh

##@ Development
config: ## Configure the build
	$(CMAKE) -S . -B $(BUILD_DIR) -G "$(GENERATOR)" $(CMAKE_CACHE_ARGS)

build: config ## Build the project
	$(CMAKE) --build $(BUILD_DIR) --parallel

##@ Testing
test: build ## Run tests
	$(CTEST) --test-dir $(BUILD_DIR) --output-on-failure

##@ Project Management
install: build ## Install the project
	$(CMAKE) --install $(BUILD_DIR)

release: ## Build release artifacts
	@echo "Building release..."

##@ Maintenance
clean: ## Remove all build artifacts
	rm -rf $(EXES) $(DIST) $(BUILD_DIR)

##@ Documentation
help: ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "Makefile Usage:\n  make \033[36m<target>\033[0m\n"} /^[.a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

