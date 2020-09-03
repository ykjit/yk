#!/bin/sh

set -e

# Rather than use the rustfmt from ykrustc, we use the last nightly snapshot
# where rustfmt worked. It's often busted for days at a time, and we don't want
# that to stall our development.
#
# We use a trick to guarantee we get a rustfmt from rustup:
# https://github.com/rust-lang/rustup/issues/2227#issuecomment-584754687
export CARGO_HOME="`pwd`/.cargo"
export RUSTUP_HOME="`pwd`/.rustup"
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > rustup.sh
sh rustup.sh --default-host x86_64-unknown-linux-gnu --default-toolchain none -y --no-modify-path
OLDPATH=${PATH}
export PATH=${CARGO_HOME}/bin/:${RUSTUP_HOME}/bin:$PATH
rustup toolchain install nightly
cargo fmt --all -- --check
export PATH=${OLDPATH}
unset CARGO_HOME
unset RUSTUP_HOME

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
# the compiler to crash if optimisations *and* a tracer are enabled.
#cargo test --release
