#!/bin/sh
#
# This is like `wrap-clang++`, just for C code.

set -e

echo "clang $@" > $(mktemp -p ${YK_CC_TEMPDIR})

clang $@
