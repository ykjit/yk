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

# Build LLVM for the C tests.
mkdir -p target && cd target
git clone https://github.com/ykjit/ykllvm
cd ykllvm
mkdir build
cd build

# Due to an LLVM bug, PIE breaks our mapper, and it's not enough to pass
# `-fno-pie` to clang for some reason:
# https://github.com/llvm/llvm-project/issues/57085
cmake -DCMAKE_INSTALL_PREFIX=`pwd`/../inst \
    -DLLVM_INSTALL_UTILS=On \
    -DCMAKE_BUILD_TYPE=release \
    -DLLVM_ENABLE_ASSERTIONS=On \
    -DLLVM_ENABLE_PROJECTS="lld;clang" \
    -DCLANG_DEFAULT_PIE_ON_LINUX=OFF \
    ../llvm
make -j `nproc` install
export PATH=`pwd`/../inst/bin:${PATH}
cd ../../..

# Check that clang-format is installed.
clang-format --version
# Check C/C++ formatting using xtask.
cargo xtask cfmt
git diff --exit-code

# Check that building `ykcapi` in isolation works. This is what we'd be doing
# if we were building release binaries, as it would mean we get a system
# without the (slower) `yk_testing` and `yk_jitstate_debug` features enabled.
for mode in "" "--release"; do
    cargo build ${mode} -p ykcapi;
done

cargo test
cargo test --release
cargo bench

# Run examples.
cargo run --example hwtracer_example
cargo run --release --example hwtracer_example
