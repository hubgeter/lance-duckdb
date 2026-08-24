# Vane uses a different DuckDB ABI and exposes the distributed execution
# contracts used by Lance. This entry point produces an independently shipped
# Vane-compatible loadable artifact; do not load it into official DuckDB (or
# load an official-DuckDB artifact into Vane). Set the mode before entering the
# shared loader so the extension's CMake is configured with the Vane headers,
# while the public entry point can still force OFF for an official build that
# reuses a cache directory.
set(LANCE_VANE_EXTENSION_CONFIG_VANE TRUE)
set(LANCE_VANE_DISTRIBUTED
    ON
    CACHE BOOL "Build Lance's Vane distributed scan and write adapters" FORCE)
include(${CMAKE_CURRENT_LIST_DIR}/extension_config.cmake)
unset(LANCE_VANE_EXTENSION_CONFIG_VANE)
