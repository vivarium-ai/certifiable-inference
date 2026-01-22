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
PREFIX ?= /usr/local

CMAKE ?= cmake
CTEST ?= ctest

# ccache (enabled by default if available)
CCACHE ?= ccache
CCACHE_DIR ?= $(HOME)/.ccache

# Export for scripts
export BUILD_DIR
export BUILD_TYPE
export GENERATOR
export PREFIX
export CMAKE
export CTEST
export CCACHE
export CCACHE_DIR

CMAKE_CACHE_ARGS := \
	-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
	-DCMAKE_C_COMPILER_LAUNCHER=$(CCACHE) \
	-DCMAKE_CXX_COMPILER_LAUNCHER=$(CCACHE)

.PHONY: all help setup config build test install package release clean

all: test

##@ Dependencies
setup: ## Setup project
	./certifiable-build/scripts/setup.sh

##@ Development
config: ## Configure the build
	./certifiable-build/scripts/config.sh

build: config ## Build the project
	./certifiable-build/scripts/build.sh

##@ Testing
test: build ## Run tests
	./certifiable-build/scripts/test.sh

##@ Project Management
install: build ## Install the project
	./certifiable-build/scripts/install.sh

package: ## Build release artifacts
	./certifiable-build/scripts/package.sh

release: ## Publish release artifacts
	./certifiable-build/scripts/release.sh

##@ Maintenance
clean: ## Remove all build artifacts
	./certifiable-build/scripts/clean.sh

##@ Documentation
help: ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "Makefile Usage:\n  make \033[36m<target>\033[0m\n"} /^[.a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

