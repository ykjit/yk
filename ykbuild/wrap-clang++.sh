#!/bin/sh
#
# A wrapper around the C++ compiler to capture compilation commands performed
# by `build.rs`. This allows us to generate a `compile_commands.json` file for
# clangd, which in turn gives us LSP support for our C++ code!

set -e

echo "${YK_COMPILER_PATH} $@" > $(mktemp -p ${YK_COMPILER_TEMPDIR})
${YK_COMPILER_PATH} $@
