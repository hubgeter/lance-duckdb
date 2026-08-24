# This file is included by DuckDB's build system. It specifies which extension
# to load. Keep this entry point free of Vane-only options so that the same
# source tree remains a normal DuckDB extension. The explicit OFF assignment
# also prevents a stale cache from silently turning an official build into a
# Vane build; extension_config_vane.cmake sets a private marker before it
# includes this file.
if(NOT DEFINED LANCE_VANE_EXTENSION_CONFIG_VANE)
  set(LANCE_VANE_DISTRIBUTED OFF CACHE BOOL
      "Build Lance's Vane distributed scan and write adapters" FORCE)
endif()
duckdb_extension_load(lance
    SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}
    LOAD_TESTS
)
