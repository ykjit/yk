#!/bin/sh

set -e

case ${STD_TRACER_MODE} in
    "sw") export RUSTFLAGS="-C tracer=sw";;
    "hw") export RUSTFLAGS="-C tracer=hw";;
    *) echo "STD_TRACER_MODE must be set to either 'hw' or 'sw'"
       exit 1;;
esac

# Use the most recent successful ykrustc build.
tar jxf /opt/ykrustc-bin-snapshots/ykrustc-${STD_TRACER_MODE}-stage2-latest.tar.bz2
export PATH=`pwd`/ykrustc-stage2-latest/bin:${PATH}

cargo fmt --all -- --check
cargo test
cargo test --release
