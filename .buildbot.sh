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

# Run rustfmt.
# Note that xtask requires us to use a nightly toolchain for this step.
rustup toolchain install nightly --allow-downgrade --component rustfmt
cargo xtask fmt --all -- --check

# Build the compiler and add it as a linked toolchain.
git clone https://github.com/softdevteam/ykrustc
# Prevent cargo walking up the filesystem and picking up the yk cargo config.
# This leads to all kinds of chaos (slow build, linkage errors).
mv .cargo/config.toml .cargo/config.toml.save
cd ykrustc
cat <<EOD >> Cargo.toml
[patch."https://github.com/softdevteam/yk"]
ykpack = { path = "../internal_ws/ykpack" }
EOD
cp .buildbot.config.toml config.toml
./x.py build --stage 1
rustup toolchain link ykrustc-stage1 `pwd`/build/x86_64-unknown-linux-gnu/stage1
cd ..
mv .cargo/config.toml.save .cargo/config.toml
rustup override set ykrustc-stage1

# Test both workspaces using the compiler we just built.
# We do this for both debug and release mode.
export RUSTFLAGS="-C tracer=${CI_TRACER_KIND} -D warnings"
for mode in "" "--release"; do
    cargo xtask test ${mode}
    # Check that extra arguments (e.g. test filtering and `--nocapture`) are
    # handled correctly.
    cargo xtask test guard ${mode} -- --nocapture | grep -v binop
    cargo xtask clean ${mode}

    # Also test the build without xtask, as that's what consumers will do.
    cargo build ${mode}
    cargo clean ${mode}
done

# Benchmarking always implies release mode.
cargo xtask bench
