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

# Build LLVM for the C tests.
mkdir -p target && cd target
git clone https://github.com/ykjit/ykllvm
cd ykllvm
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=`pwd`/../inst \
    -DLLVM_INSTALL_UTILS=On \
    -DCMAKE_BUILD_TYPE=release \
    -DLLVM_ENABLE_ASSERTIONS=On \
    -DLLVM_ENABLE_PROJECTS="lld;clang" \
    ../llvm
make -j `nproc` install
export PATH=`pwd`/../inst/bin:${PATH}
cd ../../..

cargo test
cargo test --release
cargo bench
