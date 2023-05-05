#!/bin/sh
#
# A wrapper around the C++ compiler to capture compilation commands performed
# by `build.rs`. This allows us to generate a `compile_commands.json` file for
# clangd, which in turn gives us LSP support for our C++ code!

set -e

export PATH=${DEP_YKBUILD_YKLLVM}:${PATH}

echo "clang++ $@" > $(mktemp -p ${YK_CC_TEMPDIR})

clang++ $@
