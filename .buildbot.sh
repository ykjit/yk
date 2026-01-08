#!/bin/sh

set -eu

ROOT_DIR=$(realpath $(pwd))

# What git commit hash of yklua & ykcbf will we test in buildbot?
YKLUA_REPO="https://github.com/ykjit/yklua.git"
YKLUA_COMMIT="8a288908efc4611892993a32f8ffadd08df8e8c6"
YKCBF_REPO="https://github.com/ykjit/ykcbf.git"
YKCBF_COMMIT="431b92593180e1e376d08ecf383c4a1ab8473b3d"

TRACERS="swt"

# Build yklua and run the test suite.
#
# Before calling this:
#  - `yk-config` must be in PATH.
#  - YK_BUILD_TYPE must be set.
test_yklua() {
    if [ ! -e "yklua" ]; then
        git clone --depth=1 "$YKLUA_REPO"
        cd yklua
        git fetch --depth=1 origin "$YKLUA_COMMIT"
        git checkout "$YKLUA_COMMIT"
        cd ..
    fi
    cd yklua
    make clean
    make -j $(nproc)
    cd tests
    YKD_SERIALISE_COMPILATION=1 ../src/lua -e"_U=true" all.lua
    ../src/lua -e"_U=true" all.lua
    cd ../..
}

# Check that the ykllvm commit in the submodule is from the main branch.
# Due to the way github works, this may not be the case!
#
# We avoid using hardcoded remote names like 'origin' so that this works with
# TryCI (e.g. where development clones may use 'upstream' -> `ykjit/ykllvm`).
# This will always be correct on the CI server because there will only ever be
# a single remote.
#
# We need to careful with our grep regex though, ensuring we only match
# `remote/main`, and not `remote/<other_branch>` that just so happens to have `/main`
# in its branch name.
cd ykllvm
tot=$(git rev-parse HEAD)
if ! git branch -r --contains $tot | grep -qE '^([^/]+/)?main$'; then
    echo "Error: HEAD ($tot) did not originate from ykllvm's main branch"
    exit 1
fi
cd ..

# Install rustup.
CARGO_HOME="${ROOT_DIR}/.cargo"
export CARGO_HOME
RUSTUP_HOME="${ROOT_DIR}/.rustup"
export RUSTUP_HOME
export RUSTUP_INIT_SKIP_PATH_CHECK="yes"
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > rustup.sh
sh rustup.sh --default-host x86_64-unknown-linux-gnu \
    --default-toolchain nightly \
    --no-modify-path \
    --profile default \
    -y
export PATH="${CARGO_HOME}"/bin/:"$PATH"

# Formatting problems are frequent in PRs, and easy to fix, so try and catch
# those before doing anything complicated.
cargo fmt --all -- --check

# We now need a copy of ykllvm. Building this is quite slow so if there's a
# cached version in `ykllvm_cache/` we use that. Whether we build our own or
# use a cached copy, the installed version ends up in ykllvm/inst. Notice that
# we use the release version because some of the checks we run below (e.g.
# unused warnings) otherwise run slowly.

cd ykllvm
ykllvm_hash=$(git rev-parse HEAD)
cached_ykllvm=0
if [ -f /opt/ykllvm_cache/ykllvm-release-with-assertions-"${ykllvm_hash}".tgz ]; then
    mkdir inst
    cd inst
    tar xfz /opt/ykllvm_cache/ykllvm-release-with-assertions-"${ykllvm_hash}".tgz
    # Minimally check that we can at least run `clang --version`: if we can't,
    # we assume the cached binary is too old (e.g. linking against old shared
    # objects) and that we should build our own version.
    if bin/clang --version > /dev/null; then
        cached_ykllvm=1
    else
        echo "Warning: cached ykllvm not runnable; building from scratch" > /dev/stderr
        cd ..
        rm -rf inst
    fi
fi

if [ "$cached_ykllvm" -eq 0 ]; then
    # We could let yk build two copies of LLVM, but we also want to: check that
    # YKB_YKLLVM_BIN_DIR works; and we want access to clang-format from a build
    # of LLVM. So we first build (or use a prebuilt version) of our
    # ykllvm-with-assertions, use the YKB_YKLLVM_BIN_DIR variable to have yk use
    # that, and use its clang-format.

    mkdir -p build
    cd build
    # Due to an LLVM bug, PIE breaks our mapper, and it's not enough to pass
    # `-fno-pie` to clang for some reason:
    # https://github.com/llvm/llvm-project/issues/57085
    cmake -DCMAKE_INSTALL_PREFIX="$(pwd)/../inst" \
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
fi
YKLLVM_BIN_DIR=${ROOT_DIR}/ykllvm/inst/bin
export YKB_YKLLVM_BIN_DIR="${YKLLVM_BIN_DIR}"
cd ../../

# Check C/C++ formatting.
PATH=${YKB_YKLLVM_BIN_DIR}:${PATH} cargo xtask cfmt --check

cargo install cargo-diff-tools
if [ "$CI_RUNNER" = buildbot ] ; then
    # When running under buildbot, we need to `git fetch` data from the remote if we want
    # cargo-clippy-def to work later.
    git fetch --no-recurse-submodules origin master:refs/remotes/origin/master
fi

# debug mode is very slow to run yk programs, but we don't really care about
# the buildtime, so we force `debug` builds to be built with optimisations.
# Note, this still keeps `debug_assert`s, overflow checks and the like!
cat << EOF >> Cargo.toml
[profile.dev]
opt-level = 3
codegen-units = 16
EOF

for tracer in ${TRACERS}; do
    export YKB_TRACER="${tracer}"
    # Check for annoying compiler warnings in each package.
    WARNING_DEFINES="-D unused-variables -D dead-code -D unused-imports"
    for p in $(sed -n -e '/^members =/,/^\]$/{/^members =/d;/^\]$/d;p;}' \
      Cargo.toml | \
      tr -d ' \t\",' | grep -v xtask); do
        echo "$WARNING_DEFINES" | xargs cargo rustc -p "$p" --profile check --lib --
        # For some reason, we can't do these checks on crates with binary targets.
        if [ "$p" != "ykrt" ] && [ "$p" != "tests" ]; then
            echo "$WARNING_DEFINES" | xargs cargo rustc -p "$p" --profile check --tests --
            echo "$WARNING_DEFINES" | xargs cargo rustc -p "$p" --profile check --benches --
        fi
    done
    echo "$WARNING_DEFINES" | xargs cargo rustc -p tests --profile check --bin dump_ir --
    echo "$WARNING_DEFINES" | xargs cargo rustc -p tests --profile check --bin gdb_c_test --
    echo "$WARNING_DEFINES" | xargs cargo rustc -p xtask --profile check --bin xtask --

    # Error if Clippy detects any warnings introduced in lines changed in this PR.
    cargo-clippy-diff origin/master -- --all-features --tests -- -D warnings
done

# Run the tests multiple times on swt to try and catch non-deterministic
# failures. But running everything so often is expensive, so run other tracers'
# tests just once.
echo "===> Running swt tests"
for _ in $(seq 10); do
    YKB_TRACER=swt RUST_TEST_SHUFFLE=1 cargo test
done

# Keep j2 honest w.r.t. lang_tests
YK_JITC=j2 RUST_TEST_SHUFFLE=1 cargo test lang_tests::

# test yklua/swt in debug mode.
PATH=${ROOT_DIR}/bin:${PATH} YK_BUILD_TYPE=debug YKB_TRACER=swt test_yklua

for tracer in ${TRACERS}; do
    if [ "$tracer" = "swt" ]; then
        # already tested above.
        continue
    fi
    echo "===> Running ${tracer} tests"
    RUST_TEST_SHUFFLE=1 cargo test
done

# Test with LLVM sanitisers
rustup component add rust-src
# The thread sanitiser does have false positives (albeit much reduced by `-Z
# build-std`), so we have to add a suppression file to avoid those stopping
# this script from succeeding. This does mean that we might suppress some true
# positives, but there's little we can do about that.
suppressions_path=$(mktemp)
cat << EOF > "$suppressions_path"
# Thread sanitiser doesn't know about atomic operations.
race:core::sync::atomic::atomic_
# count_to_hot_location moves something into a mutex, at which point accesses
# to it are safe, but thread sanitiser doesn't seem to pick up the link between
# the two.
race:<ykrt::location::Location>::count_to_hot_location
EOF

for tracer in $TRACERS; do
    export YKB_TRACER="${tracer}"
    cargo build
    ASAN_SYMBOLIZER_PATH="${YKLLVM_BIN_DIR}/llvm-symbolizer" \
      RUSTFLAGS="-Z sanitizer=address" \
      RUSTDOCFLAGS="-Z sanitizer=address" \
      cargo test \
      -Z build-std \
      --target x86_64-unknown-linux-gnu

    RUST_TEST_THREADS=1 \
      RUSTFLAGS="-Z sanitizer=thread" \
      RUSTDOCFLAGS="-Z sanitizer=thread" \
      TSAN_OPTIONS="suppressions=$suppressions_path" \
      cargo test \
      -Z build-std \
      --target x86_64-unknown-linux-gnu
done

# Later on we are going to need to install cargo-deny and mdbook. We kick the
# install jobs off now so that at least some work (e.g. downloading crates) can
# happen in parallel, speeding up the overall process.

cargo_deny_mdbook_tmp=$(mktemp)
( cargo install --locked cargo-deny ; cargo install --locked mdbook ) \
  >"${cargo_deny_mdbook_tmp}" 2>&1 &
cargo_deny_mdbook_pid=$!

# We now want to test building with `--release`.

for tracer in $TRACERS; do
    export YKB_TRACER="${tracer}"
    echo "===> Running ${tracer} tests"

    # The lua test harness isn't clever enough to rebuild yklua when YKB_TRACER
    # changes, so for now just nuke any existing yklua to force a rebuild.
    rm -rf target/release/yklua

    RUST_TEST_SHUFFLE=1 cargo test --release

    if [ "${tracer}" = "swt" ]; then
        # test yklua/swt in release mode.
        PATH=${ROOT_DIR}/bin:${PATH} YK_BUILD_TYPE=release YKB_TRACER=${tracer} test_yklua

        # Do a quick run of the benchmark suite as a smoke test.
        pipx install rebench
        git clone --depth 1 --recurse-submodules --shallow-submodules https://github.com/ykjit/yk-benchmarks
        cd yk-benchmarks
        ln -s ../yklua .
        sed -e 's/executions: \[Lua, YkLua\]/executions: [YkLua]/' \
            -e 's/executable: yklua/executable: lua/' \
            rebench.conf > rebench2.conf
        # Initialise extra benchmarks.
        sh setup.sh yklua/src/lua
        ~/.local/bin/rebench --quick --no-denoise -c rebench2.conf
        cd ..
    fi
done

# We want to check that the benchmarks build and run correctly, but want to
# ignore the results, so run them for the minimum possible time.
#
# Note that --profile-time doesn't work without --bench, so we have to run each
# benchmark individually.
YKB_TRACER=swt cargo bench --bench promote -- --profile-time 1
# Keep j2 honest
YK_JITC=j2 YKB_TRACER=swt cargo bench --bench promote -- --profile-time 1

# Test some BF programs.
git clone --depth=1 "$YKCBF_REPO"
cd ykcbf
git fetch --depth=1 origin "$YKCBF_COMMIT"
git checkout "$YKCBF_COMMIT"
PATH=${ROOT_DIR}/bin:${PATH} YK_BUILD_TYPE=debug make
./bf_simple_yk lang_tests/bench.bf
./bf_simple_yk lang_tests/hanoi-opt.bf
cd ..

# Check licenses.
wait "${cargo_deny_mdbook_pid}" || ( cat "${cargo_deny_mdbook_tmp}" && exit 1 )
cargo-deny check license

# Build the docs
cd docs
mdbook build
test -d book
cd ..
