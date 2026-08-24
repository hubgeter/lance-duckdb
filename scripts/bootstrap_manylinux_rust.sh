#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 lance-duckdb contributors
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

rust_version="1.94.1"
rustup_version="1.28.2"
rustup_target="x86_64-unknown-linux-gnu"
rustup_sha256="20a06e644b0d9bd2fbdbfd52d42540bdde820ea7df86e92e533c073da0cdd43c"
rustup_url="https://static.rust-lang.org/rustup/archive/${rustup_version}/${rustup_target}/rustup-init"
build_root="${LANCE_DUCKDB_BUILD_TOOLS_DIR:-/tmp/lance-duckdb-build-tools}"
rustup_init="${build_root}/rustup-init-${rustup_version}-${rustup_target}"
cargo_home="/root/.cargo"
rustup_home="/root/.rustup"

mkdir -p "$build_root"
if [[ -f "$rustup_init" ]] \
  && ! echo "${rustup_sha256}  ${rustup_init}" | sha256sum --check --status; then
  rm -f "$rustup_init"
fi

if [[ ! -f "$rustup_init" ]]; then
  curl --fail --location --retry 5 --retry-delay 2 \
    --output "${rustup_init}.part" "$rustup_url"
  mv "${rustup_init}.part" "$rustup_init"
fi
echo "${rustup_sha256}  ${rustup_init}" | sha256sum --check
chmod 0755 "$rustup_init"

CARGO_HOME="$cargo_home" RUSTUP_HOME="$rustup_home" "$rustup_init" \
  --no-modify-path \
  --profile minimal \
  --default-host "$rustup_target" \
  --default-toolchain "$rust_version" \
  -y

for executable in cargo rustc rustup; do
  ln -sfn "${cargo_home}/bin/${executable}" "/usr/local/bin/${executable}"
done

installed_rust_version="$(rustc --version | awk '{print $2}')"
installed_cargo_version="$(cargo --version | awk '{print $2}')"
if [[ "$installed_rust_version" != "$rust_version" || "$installed_cargo_version" != "$rust_version" ]]; then
  echo "Failed to install Rust and Cargo ${rust_version}" >&2
  exit 1
fi

echo "Installed Rust and Cargo ${rust_version}"
