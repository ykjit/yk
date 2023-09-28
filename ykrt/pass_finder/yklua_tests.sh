#!/bin/sh

set -e

MODE=release

if [ ! -z $PRELINK_PASSES ]; then
  echo "yk-here"
  yk-config ${MODE} --prelink-pipeline "${PRELINK_PASSES}" --cflags 
else
  yk-config ${MODE} --cflags
fi

if [ ! -z $LINKTIME_PASSES ]; then
  yk-config ${MODE} --postlink-pipeline "${LINKTIME_PASSES}" --ldflags
else
  yk-config ${MODE} --ldflags
fi

make clean && make YK_BUILD_TYPE=release

cd tests

LUA=../src/lua

for serialise in 0 1; do
    for test in api bwcoercion closure code coroutine events \
        gengc pm tpack tracegc utf8 vararg; do
        echo "### YKD_SERIALISE_COMPILATION=$serialise $test.lua ###"
        YKD_SERIALISE_COMPILATION=$serialise ${LUA} ${test}.lua
    done
done
