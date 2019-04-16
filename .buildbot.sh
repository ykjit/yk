#!/bin/sh

set -e

# Use the most recent successful ykrustc build.
tar jxf /opt/ykrustc-bin-snapshots/ykrustc-stage2-latest.tar.bz2
export PATH=`pwd`/ykrustc-stage2-latest/bin:${PATH}

cargo fmt --all -- --check
cargo test
