#!/bin/sh

set -e

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

# Check licenses.
which cargo-deny | cargo install cargo-deny
cargo-deny check license

# Build the docs
cargo install mdbook
cd docs
mdbook build
test -d book
cd ..

# We could let yk build two copies of LLVM, but we also want to: check that
# YKB_YKLLVM_BIN_DIR works; and we want access to clang-format from a build
# of LLVM. So we first build our own LLVM-with-assertions, use the
# YKB_YKLLVM_BIN_DIR variable to have yk use that, and use its
# clang-format.
mkdir -p ykllvm/build
cd ykllvm/build
# Due to an LLVM bug, PIE breaks our mapper, and it's not enough to pass
# `-fno-pie` to clang for some reason:
# https://github.com/llvm/llvm-project/issues/57085
cmake -DCMAKE_INSTALL_PREFIX=`pwd`/../inst \
    -DLLVM_INSTALL_UTILS=On \
    -DCMAKE_BUILD_TYPE=release \
    -DLLVM_ENABLE_ASSERTIONS=On \
    -DLLVM_ENABLE_PROJECTS="lld;clang" \
    -DCLANG_DEFAULT_PIE_ON_LINUX=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_C_COMPILER=/usr/bin/clang \
    -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
    -GNinja \
    ../llvm
cmake --build .
cmake --install .
export YKB_YKLLVM_BIN_DIR=`pwd`/../inst/bin
cd ../../

# Check that clang-format is installed.
PATH=${YKB_YKLLVM_BIN_DIR}:${PATH} clang-format --version
# Check C/C++ formatting using xtask.
PATH=${YKB_YKLLVM_BIN_DIR}:${PATH} cargo xtask cfmt
# This is used to check clang-tidy output, but the dirty submodule from building
# ykllvm is also shown.
# FIXME: Add build/ to .gitignore in ykllvm
git diff --exit-code --ignore-submodules

# There are some feature-gated testing/debugging switches which slow the JIT
# down a bit. Check that if we build the system without tests, those features
# are not enabled.
cargo -Z unstable-options build --build-plan -p ykcapi | \
    awk '/yk_testing/ { ec=1 } /yk_jitstate_debug/ { ec=1 } END {exit ec}'

for i in $(seq 10); do
    RUST_TEST_SHUFFLE=1 cargo test
done

# We now want to test building with `--release`, which we also take as an
# opportunity to check that yk can build ykllvm, which requires unsetting
# YKB_YKLLVM_BIN_DIR. In essence, we now repeat much of what we did above but
# with `--release`.
unset YKB_YKLLVM_BIN_DIR
export YKB_YKLLVM_BUILD_ARGS="define:CMAKE_C_COMPILER=/usr/bin/clang,define:CMAKE_CXX_COMPILER=/usr/bin/clang++"

cargo -Z unstable-options build --release --build-plan -p ykcapi | \
    awk '/yk_testing/ { ec=1 } /yk_jitstate_debug/ { ec=1 } END {exit ec}'

cargo build --release -p ykcapi

for i in $(seq 10); do
    RUST_TEST_SHUFFLE=1 cargo test --release
done

cargo bench
