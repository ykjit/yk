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
YKLLVM_BIN_DIR=$(pwd)/../inst/bin
export YKB_YKLLVM_BIN_DIR=${YKLLVM_BIN_DIR}
cd ../../

# Check that clang-format is installed.
PATH=${YKB_YKLLVM_BIN_DIR}:${PATH} clang-format --version
# Check C/C++ formatting using xtask.
PATH=${YKB_YKLLVM_BIN_DIR}:${PATH} cargo xtask cfmt
# This is used to check clang-tidy output, but the dirty submodule from building
# ykllvm is also shown.
# FIXME: Add build/ to .gitignore in ykllvm
git diff --exit-code --ignore-submodules

# Check for annoying compiler warnings in each package.
WARNING_DEFINES="-D unused-variables -D dead-code -D unused-imports"
for p in $(sed -n -e '/^members =/,/^\]$/{/^members =/d;/^\]$/d;p;}' \
  Cargo.toml \
  | \
  tr -d ' \t\",' | grep -v xtask); do
    cargo rustc -p $p --profile check --lib -- ${WARNING_DEFINES}
    # For some reason, we can't do these checks on crates with binary targets.
    if [ "$p" != "ykrt" ] && [ "$p" != "tests" ]; then
        cargo rustc -p $p --profile check --tests -- ${WARNING_DEFINES}
        cargo rustc -p $p --profile check --benches -- ${WARNING_DEFINES}
    fi
done
cargo rustc -p tests --profile check --bin dump_ir -- ${WARNING_DEFINES}
cargo rustc -p tests --profile check --bin gdb_c_test -- ${WARNING_DEFINES}
cargo rustc -p xtask --profile check --bin xtask -- ${WARNING_DEFINES}

# There are some feature-gated testing/debugging switches which slow the JIT
# down a bit. Check that if we build the system without tests, those features
# are not enabled.
cargo -Z unstable-options build --build-plan -p ykcapi | \
    awk '/yk_testing/ { ec=1 } END {exit ec}'
cargo -Z unstable-options build --build-plan -p ykrt | \
    awk '/yk_testing/ { ec=1 } /yk_jitstate_debug/ { ec=1 } END {exit ec}'

TRACERS="hwt swt"

for tracer in $TRACERS; do 
    echo "Running ${tracer} tests"
    for i in $(seq 10); do
        YKB_TRACER="${tracer}" RUST_TEST_SHUFFLE=1 cargo test
        YKB_TRACER="${tracer}" YKD_NEW_CODEGEN=1 RUST_TEST_SHUFFLE=1 cargo test
    done
done

# Test with LLVM sanitisers
rustup component add rust-src
for tracer in $TRACERS; do 
    YKB_TRACER="${tracer}" cargo build
    YKB_TRACER="${tracer}" ASAN_SYMBOLIZER_PATH=${YKLLVM_BIN_DIR}/llvm-symbolizer \
        RUSTFLAGS="-Z sanitizer=address" cargo test \
        -Z build-std \
        --target x86_64-unknown-linux-gnu
done
# The thread sanitiser does have false positives (albeit much reduced by `-Z
# build-std`), so we have to add a suppression file to avoid those stopping
# this script from succeeding. This does mean that we might suppress some true
# positives, but there's little we can do about that.
suppressions_path=`mktemp`
cat << EOF > $suppressions_path
# Thread sanitiser doesn't know about atomic operations.
race:core::sync::atomic::atomic_
# count_to_hot_location moves something into a mutex, at which point accesses
# to it are safe, but thread sanitiser doesn't seem to pick up the link between
# the two.
race:ykrt::location::Location::count_to_hot_location
EOF
for tracer in $TRACERS; do 
    YKB_TRACER="${tracer}" cargo build
    YKB_TRACER="${tracer}" RUST_TEST_THREADS=1 \
        RUSTFLAGS="-Z sanitizer=thread" \
        TSAN_OPTIONS="suppressions=$suppressions_path" \
        cargo test \
        -Z build-std \
        --target x86_64-unknown-linux-gnu
done
# We now want to test building with `--release`, which we also take as an
# opportunity to check that yk can build ykllvm, which requires unsetting
# YKB_YKLLVM_BIN_DIR. In essence, we now repeat much of what we did above but
# with `--release`.
unset YKB_YKLLVM_BIN_DIR
export YKB_YKLLVM_BUILD_ARGS="define:CMAKE_C_COMPILER=/usr/bin/clang,define:CMAKE_CXX_COMPILER=/usr/bin/clang++"

cargo -Z unstable-options build --release --build-plan -p ykcapi | \
    awk '/yk_testing/ { ec=1 } /yk_jitstate_debug/ { ec=1 } END {exit ec}'

for tracer in $TRACERS; do 
    YKB_TRACER="${tracer}" cargo build --release -p ykcapi
    echo "Running ${tracer} tests"
    for i in $(seq 10); do
        YKB_TRACER="${tracer}" RUST_TEST_SHUFFLE=1 cargo test --release
        YKB_TRACER="${tracer}" YKD_NEW_CODEGEN=1 RUST_TEST_SHUFFLE=1 cargo test --release
    done
done

# We want to check that the benchmarks build and run correctly, but want to
# ignore the results, so run them for the minimum possible time.
#
# Note that --profile-time doesn't work without --bench, so we have to run each
# benchmark individually.
for b in collect_and_decode promote; do
    cargo bench --bench ${b} -- --profile-time 1
done
