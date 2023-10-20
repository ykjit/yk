#!/bin/sh
# This script runs tests cases for YK and YKLUA. The aim is to feed this test
# script to try_passes.py and get list of passes which are successful for all
# test suites. We can expand this script to accomodate new test suites in
# future.

# Check if YK_PATH is set and is a valid directory.
if [ -z "$YK_PATH" ] || [ ! -d "$YK_PATH" ]; then
    echo "YK_PATH directory does not exist."
    exit 1
fi

# Move to YK_PATH and run cargo test.
cd "$YK_PATH"
cargo test --release
cargo test --debug

# Check if YKLUA_PATH is set and is a valid directory.
if [ -z "$YKLUA_PATH" ] || [ ! -d "$YKLUA_PATH" ]; then
    echo "YKLUA_PATH directory does not exist."
    exit 1
fi

cd $YKLUA_PATH
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
