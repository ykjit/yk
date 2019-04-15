#!/bin/sh

set -e

# First we have to build ykrustc, complete with cargo and rustfmt.
ykrustc_prefix=`pwd`/ykrustc-inst
ykrustc_version=master

if ! [ -d ykrustc ]; then
    git clone https://github.com/softdevteam/ykrustc
fi
cd ykrustc
git checkout ${ykrustc_version}

cat << EOF > config.toml
[build]
extended = true
tools = ["cargo", "rustfmt"]

[install]
prefix = "${ykrustc_prefix}"
sysconfdir = "etc"

[rust]
codegen-units = 0
debug-assertions = true

[llvm]
assertions = true
EOF

mkdir -p ${ykrustc_prefix}

if ! [ -e "${ykrustc_prefix}/bin/rustc" ]; then
    ./x.py install
fi

export PATH=${ykrustc_prefix}/bin:${PATH}
cd ..

# Now we test yktrace with ykrustc.
cargo test
cargo fmt -- --check
