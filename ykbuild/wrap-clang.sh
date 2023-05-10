#!/bin/sh
#
# This is like `wrap-clang++`, just for C code.

set -e

echo dep: ${DEP_YKBUILD_YKLLVM}
echo path: $PATH
export PATH=${DEP_YKBUILD_YKLLVM}:${PATH}

echo "clang $@" > $(mktemp -p ${YK_CC_TEMPDIR})

clang $@
