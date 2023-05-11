#!/bin/sh
#
# This is like `wrap-clang++`, just for C code.

set -e

echo "${YK_COMPILER_PATH} $@" > $(mktemp -p ${YK_COMPILER_TEMPDIR})
${YK_COMPILER_PATH} $@
