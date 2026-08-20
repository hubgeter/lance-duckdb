cmake_minimum_required(VERSION 3.5)

# Set extension name here
set(TARGET_NAME lance)

set(EXTENSION_NAME ${TARGET_NAME}_extension)

project(${TARGET_NAME})
include_directories(src/include)

set(EXTENSION_SOURCES
    src/lance_extension.cpp
    src/lance_scan.cpp
    src/lance_search.cpp
    src/lance_common.cpp
    src/lance_dataset_cache.cpp
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
# Vane embeds Lance in vane._native.  Deliberately do not produce a loadable
# DuckDB extension: it would carry a second runtime entry point and could be
# loaded into a DuckDB binary with a different C++ ABI.
set_target_properties(${EXTENSION_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

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
set(RUST_FFI_DEPENDS ${CMAKE_CURRENT_LIST_DIR}/Cargo.toml
                     ${CMAKE_CURRENT_LIST_DIR}/Cargo.lock
                     ${CMAKE_CURRENT_LIST_DIR}/rust-toolchain.toml
                     ${CMAKE_CURRENT_LIST_DIR}/.cargo/config.toml
                     ${CMAKE_CURRENT_LIST_DIR}/rust/ffi_guard_macro/Cargo.toml
                     ${RUST_FFI_SOURCE_DEPENDS})

set(RUST_CARGO_ENV CARGO_TARGET_DIR=${RUST_TARGET_DIR}
                   MACOSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET})
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
  if(DEFINED VCPKG_TARGET_TRIPLET)
    set(RUST_PROTOC_TRIPLET "${VCPKG_TARGET_TRIPLET}")
  elseif(RUST_PLATFORM_TARGET MATCHES "^x86_64-.*-linux")
    set(RUST_PROTOC_TRIPLET "x64-linux")
  elseif(RUST_PLATFORM_TARGET MATCHES "^aarch64-.*-linux")
    set(RUST_PROTOC_TRIPLET "arm64-linux")
  elseif(RUST_PLATFORM_TARGET STREQUAL "x86_64-apple-darwin")
    set(RUST_PROTOC_TRIPLET "x64-osx")
  elseif(RUST_PLATFORM_TARGET STREQUAL "aarch64-apple-darwin")
    set(RUST_PROTOC_TRIPLET "arm64-osx")
  elseif(RUST_PLATFORM_TARGET MATCHES "^x86_64-.*-windows")
    set(RUST_PROTOC_TRIPLET "x64-windows")
  elseif(RUST_PLATFORM_TARGET MATCHES "^aarch64-.*-windows")
    set(RUST_PROTOC_TRIPLET "arm64-windows")
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

  # The Python build configures Vane itself as the CMake source tree, so its
  # already-installed vcpkg host tools are available even when the extension
  # variables above are not exported into this directory.
  unset(RUST_PROTOC_PATH CACHE)
  find_program(
    RUST_PROTOC_PATH
    NAMES protoc protoc.exe
    HINTS ${RUST_PROTOC_HINTS})
  if(NOT RUST_PROTOC_PATH)
    message(
      FATAL_ERROR
        "protoc is required to build the lance-duckdb Rust library")
  endif()
  list(APPEND RUST_CARGO_ENV "PROTOC=${RUST_PROTOC_PATH}")
endif()

# Building both debug and release Rust artifacts is expensive and can trigger
# OOMs in CI. For single-config generators (Ninja/Makefiles), only build the
# active configuration's artifact.
if(CMAKE_CONFIGURATION_TYPES)
  # Multi-config generators select one configuration at build time. A custom
  # target is intentional here: Cargo's own dependency tracking makes repeat
  # invocations cheap, while building both profiles for every configuration
  # can double peak disk and memory use.
  add_custom_target(
    lance_duckdb_ffi_build
    COMMAND
      ${CMAKE_COMMAND} -E env ${RUST_CARGO_ENV} cargo build --locked
      --no-default-features --manifest-path=${CMAKE_CURRENT_LIST_DIR}/Cargo.toml
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
        --no-default-features
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
        --no-default-features
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
add_dependencies(${EXTENSION_NAME} lance_duckdb_ffi_build)

install(
  TARGETS ${EXTENSION_NAME}
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
