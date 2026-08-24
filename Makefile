PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

EXT_NAME=lance
EXT_CONFIG=${PROJ_DIR}extension_config.cmake
CORE_EXTENSIONS=''

ifeq (,$(findstring -DSKIP_EXTENSIONS=,$(EXT_FLAGS)))
	EXT_FLAGS += -DSKIP_EXTENSIONS=parquet
endif
# The ordinary DuckDB targets remain available when the optional Vane tooling
# submodule has not been checked out.
include extension-ci-tools/makefiles/duckdb_extension.Makefile
-include vane-extension-ci-tools/makefiles/vane_extension.Makefile

.PHONY: configure_ci vane_configure_ci
configure_ci: vane_configure_ci

vane_configure_ci:
	@bash scripts/configure_ci.sh
