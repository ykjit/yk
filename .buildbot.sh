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

# There are some feature-gated testing/debugging switches which slow the JIT
# down a bit. Check that if we build the system without tests, those features
# are not enabled.
for mode in "" "--release"; do \
    cargo -Z unstable-options build ${mode} --build-plan -p ykcapi | \
        awk '/yk_testing/ { ec=1 } /yk_jitstate_debug/ { ec=1 } END {exit ec}'; \
done

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

# We build our own LLVM-with-assertions to get access to clang-format. Since
# we're going to such lengths, we then reuse this installation of LLVM when
# doing non-`--build` releases below.
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
    -GNinja \
    ../llvm
cmake --build .
cmake --install .
export YKB_YKLLVM_INSTALL_DIR=`pwd`/../inst/bin
cd ../../

# Check that clang-format is installed.
PATH=${YKB_YKLLVM_INSTALL_DIR}:${PATH} clang-format --version
# Check C/C++ formatting using xtask.
PATH=${YKB_YKLLVM_INSTALL_DIR}:${PATH} cargo xtask cfmt

# This is used to check clang-tidy output, but the dirty submodule from building
# ykllvm is also shown.
# FIXME: Add build/ to .gitignore in ykllvm
git diff --exit-code --ignore-submodules

# Check that building `ykcapi` in isolation works. This is what we'd be doing
# if we were building release binaries, as it would mean we get a system
# without the (slower) `yk_testing` and `yk_jitstate_debug` features enabled.
for mode in "" "--release"; do
    cargo build ${mode} -p ykcapi;
done

for i in $(seq 10); do
    cargo test
    cargo test --release
done

# Run examples.
cargo run --example hwtracer_example
cargo run --release --example hwtracer_example

# Run cargo bench, forcing yk to build its own LLVM-without-assertions.
unset YKB_YKLLVM_INSTALL_DIR
cargo bench
