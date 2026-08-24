# CONFIGURE_DEPENDS below requires CMake 3.12. Keep the declared minimum in sync
# with the commands used by this build rather than relying on a newer parent
# project to mask the requirement.
cmake_minimum_required(VERSION 3.12)

# Set extension name here
set(TARGET_NAME lance)

set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

project(${TARGET_NAME})
include_directories(src/include)

option(LANCE_VANE_DISTRIBUTED
       "Build the explicit Vane distributed scan and write interfaces" OFF)

# Vane's fork contains DuckDB's serialized LogicalWriteTarget contract.  The
# official DuckDB tree used by the loadable extension may predate that contract;
# the extension remains local-only there and must not depend on the Vane-only
# CreateTableInfo/TableCatalogEntry fields.
if(EXISTS
   "${DUCKDB_SOURCE_PATH}/src/include/duckdb/planner/logical_write_target.hpp")
  add_definitions(-DLANCE_DUCKDB_HAS_LOGICAL_WRITE_TARGET)
endif()

if(LANCE_VANE_DISTRIBUTED)
  if(NOT
     EXISTS
     "${DUCKDB_SOURCE_PATH}/src/include/duckdb/function/distributed_table_function.hpp"
  )
    message(
      FATAL_ERROR "LANCE_VANE_DISTRIBUTED requires the Vane DuckDB source tree")
  endif()
  if(NOT EXISTS
     "${DUCKDB_SOURCE_PATH}/src/include/duckdb/function/distributed_write.hpp")
    message(
      FATAL_ERROR
        "LANCE_VANE_DISTRIBUTED requires Vane distributed write headers")
  endif()
  add_definitions(-DLANCE_VANE_DISTRIBUTED)
endif()

set(EXTENSION_SOURCES
    src/lance_extension.cpp
    src/lance_scan.cpp
    src/lance_search.cpp
    src/lance_common.cpp
    src/lance_dataset_cache.cpp
    src/lance_lease.cpp
    src/lance_session_state.cpp
    src/lance_secrets.cpp
    src/lance_filter_ir.cpp
    src/lance_exec_ir.cpp
    src/lance_logical_exec.cpp
    src/lance_storage.cpp
    src/lance_metadata.cpp
    src/lance_maintenance.cpp
    src/lance_replacement.cpp
    src/lance_insert.cpp
    src/lance_merge.cpp
    src/lance_delete.cpp
    src/lance_update.cpp
    src/lance_expr_ir.cpp
    src/lance_write.cpp
    src/lance_truncate.cpp
    src/lance_index.cpp
    src/lance_arrow_compat.cpp
    src/lance_resolver.cpp)

build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
# The static target exists only for DuckDB's native unittest executable. The
# independently distributed product is the loadable target in both ABI modes.
# Each target is built against the one DuckDB source tree selected by the
# enclosing build; official-DuckDB and Vane artifacts are not ABI-compatible.
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})
set_target_properties(${EXTENSION_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(TARGET ${LOADABLE_EXTENSION_NAME})
  set_target_properties(${LOADABLE_EXTENSION_NAME}
                        PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

# Platform libs required by Rust deps (e.g., TLS, timezone)
if(APPLE)
  if(NOT DEFINED CMAKE_OSX_DEPLOYMENT_TARGET OR CMAKE_OSX_DEPLOYMENT_TARGET
                                                STREQUAL "")
    set(CMAKE_OSX_DEPLOYMENT_TARGET
        13.3
        CACHE STRING "Minimum OS X deployment version" FORCE)
  endif()
  set(PLATFORM_LIBS "-framework CoreFoundation" "-framework IOKit"
                    "-framework SystemConfiguration" "-framework Security")
elseif(UNIX)
  set(PLATFORM_LIBS m c)
elseif(WIN32)
  set(PLATFORM_LIBS
      ws2_32
      userenv
      advapi32
      crypt32
      ntdll
      bcrypt
      ncrypt
      secur32)
endif()

# Build and link Rust staticlib (lance_duckdb_ffi)
set(RUST_PLATFORM_TARGET "")

set(DUCKDB_PLATFORM "$ENV{DUCKDB_PLATFORM}")
if(DUCKDB_PLATFORM STREQUAL "linux_amd64_musl")
  set(RUST_PLATFORM_TARGET "x86_64-unknown-linux-musl")
elseif(DUCKDB_PLATFORM STREQUAL "linux_arm64_musl")
  set(RUST_PLATFORM_TARGET "aarch64-unknown-linux-musl")
elseif(DUCKDB_PLATFORM STREQUAL "windows_amd64_mingw")
  set(RUST_PLATFORM_TARGET "x86_64-pc-windows-gnu")
endif()

if(RUST_PLATFORM_TARGET STREQUAL "")
  if("${OS_NAME}" STREQUAL "linux")
    if("${OS_ARCH}" STREQUAL "arm64")
      set(RUST_PLATFORM_TARGET "aarch64-unknown-linux-gnu")
    else()
      set(RUST_PLATFORM_TARGET "x86_64-unknown-linux-gnu")
    endif()
  elseif("${OS_NAME}" STREQUAL "osx")
    if("${OSX_BUILD_ARCH}" STREQUAL "arm64")
      set(RUST_PLATFORM_TARGET "aarch64-apple-darwin")
    elseif("${OSX_BUILD_ARCH}" STREQUAL "x86_64")
      set(RUST_PLATFORM_TARGET "x86_64-apple-darwin")
    elseif("${OS_ARCH}" STREQUAL "arm64")
      set(RUST_PLATFORM_TARGET "aarch64-apple-darwin")
    else()
      set(RUST_PLATFORM_TARGET "x86_64-apple-darwin")
    endif()
  elseif("${OS_NAME}" STREQUAL "windows")
    if("${OS_ARCH}" STREQUAL "arm64")
      set(RUST_PLATFORM_TARGET "aarch64-pc-windows-msvc")
    else()
      set(RUST_PLATFORM_TARGET "x86_64-pc-windows-msvc")
    endif()
  endif()
endif()

if(RUST_PLATFORM_TARGET STREQUAL "")
  if(APPLE)
    if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
      set(RUST_PLATFORM_TARGET "aarch64-apple-darwin")
    else()
      set(RUST_PLATFORM_TARGET "x86_64-apple-darwin")
    endif()
  elseif(UNIX)
    if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
      set(RUST_PLATFORM_TARGET "aarch64-unknown-linux-gnu")
    else()
      set(RUST_PLATFORM_TARGET "x86_64-unknown-linux-gnu")
    endif()
  elseif(WIN32)
    set(RUST_PLATFORM_TARGET "x86_64-pc-windows-msvc")
  endif()
endif()

if(WIN32)
  if(RUST_PLATFORM_TARGET MATCHES "windows-gnu")
    set(RUST_STATICLIB_NAME "liblance_duckdb_ffi.a")
  else()
    set(RUST_STATICLIB_NAME "lance_duckdb_ffi.lib")
  endif()
else()
  set(RUST_STATICLIB_NAME "liblance_duckdb_ffi.a")
endif()

set(RUST_TARGET_DIR "${CMAKE_BINARY_DIR}/cargo")
set(RUST_DEBUG_LIB
    "${RUST_TARGET_DIR}/${RUST_PLATFORM_TARGET}/debug/${RUST_STATICLIB_NAME}")
set(RUST_RELEASE_LIB
    "${RUST_TARGET_DIR}/${RUST_PLATFORM_TARGET}/release/${RUST_STATICLIB_NAME}")

file(GLOB_RECURSE RUST_FFI_SOURCE_DEPENDS CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_LIST_DIR}/rust/*.rs")
set(RUST_FFI_DEPENDS
    ${CMAKE_CURRENT_LIST_DIR}/Cargo.toml
    ${CMAKE_CURRENT_LIST_DIR}/Cargo.lock
    ${CMAKE_CURRENT_LIST_DIR}/rust-toolchain.toml
    ${CMAKE_CURRENT_LIST_DIR}/.cargo/config.toml
    ${CMAKE_CURRENT_LIST_DIR}/rust/ffi_guard_macro/Cargo.toml
    ${RUST_FFI_SOURCE_DEPENDS})

set(RUST_CARGO_ENV CARGO_TARGET_DIR=${RUST_TARGET_DIR}
                   MACOSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET})
set(RUST_CARGO_FEATURE_ARGS)
if(LANCE_VANE_DISTRIBUTED)
  list(APPEND RUST_CARGO_FEATURE_ARGS --features vane-distributed)
endif()
if(DEFINED VCPKG_ROOT)
  list(APPEND RUST_CARGO_ENV VCPKG_ROOT=${VCPKG_ROOT})
endif()
if(DEFINED VCPKG_INSTALLED_DIR)
  list(APPEND RUST_CARGO_ENV VCPKG_INSTALLED_DIR=${VCPKG_INSTALLED_DIR})
endif()
if(DEFINED VCPKG_HOST_TRIPLET)
  list(APPEND RUST_CARGO_ENV VCPKG_HOST_TRIPLET=${VCPKG_HOST_TRIPLET})
endif()
if(DEFINED VCPKG_TARGET_TRIPLET)
  list(APPEND RUST_CARGO_ENV VCPKG_TARGET_TRIPLET=${VCPKG_TARGET_TRIPLET})
endif()
# prost-build needs an explicit PROTOC in DuckDB core extension builds because
# the helper script that populates CI env vars is not part of that path.
if(NOT "$ENV{PROTOC}" STREQUAL "")
  list(APPEND RUST_CARGO_ENV "PROTOC=$ENV{PROTOC}")
else()
  # protoc executes on the build host. Prefer the vcpkg host triplet during
  # cross-compilation; the target triplet may contain a non-runnable binary.
  if(DEFINED VCPKG_HOST_TRIPLET)
    set(RUST_PROTOC_TRIPLET "${VCPKG_HOST_TRIPLET}")
  else()
    # A compiler target is not necessarily executable on the build host. Use
    # CMake's host identity when vcpkg did not publish an explicit host triplet.
    string(TOLOWER "${CMAKE_HOST_SYSTEM_PROCESSOR}" RUST_PROTOC_HOST_PROCESSOR)
    if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
      if(RUST_PROTOC_HOST_PROCESSOR MATCHES "^(x86_64|amd64)$")
        set(RUST_PROTOC_TRIPLET "x64-linux")
      elseif(RUST_PROTOC_HOST_PROCESSOR MATCHES "^(aarch64|arm64)$")
        set(RUST_PROTOC_TRIPLET "arm64-linux")
      endif()
    elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
      if(RUST_PROTOC_HOST_PROCESSOR MATCHES "^(x86_64|amd64)$")
        set(RUST_PROTOC_TRIPLET "x64-osx")
      elseif(RUST_PROTOC_HOST_PROCESSOR MATCHES "^(aarch64|arm64)$")
        set(RUST_PROTOC_TRIPLET "arm64-osx")
      endif()
    elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
      if(RUST_PROTOC_HOST_PROCESSOR MATCHES "^(x86_64|amd64)$")
        set(RUST_PROTOC_TRIPLET "x64-windows")
      elseif(RUST_PROTOC_HOST_PROCESSOR MATCHES "^(aarch64|arm64)$")
        set(RUST_PROTOC_TRIPLET "arm64-windows")
      endif()
    endif()
  endif()

  set(RUST_PROTOC_HINTS "")
  if(DEFINED RUST_PROTOC_TRIPLET)
    if(DEFINED VCPKG_INSTALLED_DIR)
      list(APPEND RUST_PROTOC_HINTS
           "${VCPKG_INSTALLED_DIR}/${RUST_PROTOC_TRIPLET}/tools/protobuf")
    endif()
    if(DEFINED _VCPKG_INSTALLED_DIR)
      list(APPEND RUST_PROTOC_HINTS
           "${_VCPKG_INSTALLED_DIR}/${RUST_PROTOC_TRIPLET}/tools/protobuf")
    endif()
    list(
      APPEND
      RUST_PROTOC_HINTS
      "${CMAKE_SOURCE_DIR}/vcpkg_installed/${RUST_PROTOC_TRIPLET}/tools/protobuf"
      "${CMAKE_BINARY_DIR}/vcpkg_installed/${RUST_PROTOC_TRIPLET}/tools/protobuf"
    )
  endif()

  # A Vane-compatible out-of-tree build can reuse host tools from the selected
  # Vane dependency prefix when the extension variables are not exported.
  unset(RUST_PROTOC_PATH CACHE)
  find_program(
    RUST_PROTOC_PATH
    NAMES protoc protoc.exe
    HINTS ${RUST_PROTOC_HINTS})
  if(RUST_PROTOC_PATH)
    list(APPEND RUST_CARGO_ENV "PROTOC=${RUST_PROTOC_PATH}")
  else()
    # Configuration-only consumers such as clang-tidy do not build the Rust
    # target and therefore do not need protoc. A real Cargo build still fails
    # closed in prost-build when neither PROTOC nor a system protoc exists.
    message(STATUS "protoc was not found; it is required when building Lance")
  endif()
endif()

# Building both debug and release Rust artifacts is expensive and can trigger
# OOMs in CI. For single-config generators (Ninja/Makefiles), only build the
# active configuration's artifact.
if(CMAKE_CONFIGURATION_TYPES)
  # Multi-config generators select one configuration at build time. A custom
  # target is intentional here: Cargo's own dependency tracking makes repeat
  # invocations cheap, while building both profiles for every configuration can
  # double peak disk and memory use.
  add_custom_target(
    lance_duckdb_ffi_build
    COMMAND
      ${CMAKE_COMMAND} -E env ${RUST_CARGO_ENV} cargo build --locked
      ${RUST_CARGO_FEATURE_ARGS}
      --manifest-path=${CMAKE_CURRENT_LIST_DIR}/Cargo.toml
      --profile=$<$<CONFIG:Debug>:dev>$<$<NOT:$<CONFIG:Debug>>:release>
      --target=${RUST_PLATFORM_TARGET}
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
    DEPENDS ${RUST_FFI_DEPENDS}
    VERBATIM)
else()
  if(NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE Release)
  endif()

  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_custom_command(
      OUTPUT "${RUST_DEBUG_LIB}"
      COMMAND
        ${CMAKE_COMMAND} -E env ${RUST_CARGO_ENV} cargo build --locked
        ${RUST_CARGO_FEATURE_ARGS}
        --manifest-path=${CMAKE_CURRENT_LIST_DIR}/Cargo.toml
        --target=${RUST_PLATFORM_TARGET}
      WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
      DEPENDS ${RUST_FFI_DEPENDS}
      VERBATIM)

    add_custom_target(lance_duckdb_ffi_build DEPENDS "${RUST_DEBUG_LIB}")
  else()
    add_custom_command(
      OUTPUT "${RUST_RELEASE_LIB}"
      COMMAND
        ${CMAKE_COMMAND} -E env ${RUST_CARGO_ENV} cargo build --locked
        ${RUST_CARGO_FEATURE_ARGS}
        --manifest-path=${CMAKE_CURRENT_LIST_DIR}/Cargo.toml --release
        --target=${RUST_PLATFORM_TARGET}
      WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
      DEPENDS ${RUST_FFI_DEPENDS}
      VERBATIM)

    add_custom_target(lance_duckdb_ffi_build DEPENDS "${RUST_RELEASE_LIB}")
  endif()
endif()

target_link_libraries(${EXTENSION_NAME} debug "${RUST_DEBUG_LIB}" optimized
                      "${RUST_RELEASE_LIB}" ${PLATFORM_LIBS})
if(TARGET ${LOADABLE_EXTENSION_NAME})
  target_link_libraries(${LOADABLE_EXTENSION_NAME} debug "${RUST_DEBUG_LIB}"
                        optimized "${RUST_RELEASE_LIB}" ${PLATFORM_LIBS})
endif()
add_dependencies(${EXTENSION_NAME} lance_duckdb_ffi_build)
if(TARGET ${LOADABLE_EXTENSION_NAME})
  add_dependencies(${LOADABLE_EXTENSION_NAME} lance_duckdb_ffi_build)
endif()

set(LANCE_INSTALL_TARGETS ${EXTENSION_NAME})
if(TARGET ${LOADABLE_EXTENSION_NAME})
  list(APPEND LANCE_INSTALL_TARGETS ${LOADABLE_EXTENSION_NAME})
endif()
install(
  TARGETS ${LANCE_INSTALL_TARGETS}
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
