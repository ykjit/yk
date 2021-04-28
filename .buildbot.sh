#!/bin/sh

set -e

case ${CI_TRACER_KIND} in
    "sw" | "hw" ) true;;
    *) echo "CI_TRACER_KIND must be set to either 'hw' or 'sw'"
       exit 1;;
esac

# Install rustup.
export CARGO_HOME="`pwd`/.cargo"
export RUSTUP_HOME="`pwd`/.rustup"
export RUSTUP_INIT_SKIP_PATH_CHECK="yes"
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > rustup.sh
sh rustup.sh --default-host x86_64-unknown-linux-gnu \
    --default-toolchain nightly \
    --no-modify-path \
    --profile minimal \
    -y
export PATH=${CARGO_HOME}/bin/:$PATH

rustup toolchain install nightly --allow-downgrade --component rustfmt

cargo fmt --all -- --check

# Run the Rust tests.
cargo test
cargo test --release
cargo bench

# Build LLVM for the C tests.
#
# This is required because we have an un-upstreamed patch to get the post-LTO
# blockmap section into the end binaries.
#
# Also note that this is a fork of Rust's fork, as we hope to get all of this
# working for Rust binaries one day. Blocker:
# https://github.com/rust-lang/rust/issues/84395
cd target
git clone -b yk/12.0-2021-04-15 https://github.com/vext01/llvm-project
cd llvm-project
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=`pwd`/inst \
    -DLLVM_INSTALL_UTILS=On \
    -DCMAKE_BUILD_TYPE=release \
    -DLLVM_ENABLE_ASSERTIONS=On \
    -DLLVM_ENABLE_PROJECTS="lld;clang" \
    ../llvm
make -j `nproc` install
export PATH=`pwd`/inst/bin:${PATH}
cd ../../..

# Run the C tests.
make
