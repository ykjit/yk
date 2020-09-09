#!/bin/sh

set -e

case ${STD_TRACER_MODE} in
    "sw") export RUSTFLAGS="-C tracer=sw -Dwarnings";;
    "hw") export RUSTFLAGS="-C tracer=hw -Dwarnings";;
    *) echo "STD_TRACER_MODE must be set to either 'hw' or 'sw'"
       exit 1;;
esac

# Use the most recent successful ykrustc build.
tar jxf /opt/ykrustc-bin-snapshots/ykrustc-${STD_TRACER_MODE}-stage2-latest.tar.bz2
export PATH=`pwd`/ykrustc-stage2-latest/bin:${PATH}

cargo test
# Although it might be tempting to test release mode, we have (for now) made
# the compiler crash if optimisations *and* a tracer are enabled.
#cargo test --release

unset RUSTFLAGS
export CARGO_HOME="`pwd`/.cargo"
export RUSTUP_HOME="`pwd`/.rustup"
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > rustup.sh
sh rustup.sh --default-host x86_64-unknown-linux-gnu \
    --default-toolchain nightly \
    --no-modify-path \
    --profile minimal \
    -y
export PATH=${CARGO_HOME}/bin/:$PATH
rustup toolchain install nightly --allow-downgrade --component rustfmt
cargo +nightly fmt --all -- --check
