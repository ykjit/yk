#!/bin/sh

TIMEOUT=60
export LOGFILE="/home/shreei/research/yk_pv/ykrt/pass_finder/run_tests.log"
# Check if the LOGFILE exists and delete it before proceeding.
if [ -f "$LOGFILE" ]; then
    rm "$LOGFILE"
fi

# Check if YK_PATH is set and is a valid directory.
if [ -z "$YK_PATH" ] || [ ! -d "$YK_PATH" ]; then
    echo "YK_PATH directory does not exist." | tee -a "$LOGFILE"
    exit 1
fi

# Move to YK_PATH and run cargo test.
cd "$YK_PATH"

python3 cargo_run.py 
ret_code=$? # Capture the exit code of the python command
echo $ret_code
# Check the return code of the Python script
if [ $ret_code -eq 0 ]; then
    # If return code is zero, grep for the last line of the output
    mean_time=$(echo "$output" | tail -n 1)
    echo "$mean_time"
else
   exit $ret_code
fi 

# awk '/real/ {print $2}' $LOGFILE

# # Check if real_value is not empty and echo it, else echo 0
# if [ -n "$real_value" ]; then
#     echo "$real_value"
# else
#     echo 0
# fi
#
# # Check if YKLUA_PATH is set and is a valid directory.
# if [ -z "$YKLUA_PATH" ] || [ ! -d "$YKLUA_PATH" ]; then
#     echo "YKLUA_PATH directory does not exist."
#     exit 1
# fi
#
# cd $YKLUA_PATH
# MODE=release
#
# if [ ! -z $PRELINK_PASSES ]; then
#   echo "yk-here"
#   yk-config ${MODE} --prelink-pipeline "${PRELINK_PASSES}" --cflags 
# else
#   yk-config ${MODE} --cflags
# fi
#
# if [ ! -z $LINKTIME_PASSES ]; then
#   yk-config ${MODE} --postlink-pipeline "${LINKTIME_PASSES}" --ldflags
# else
#   yk-config ${MODE} --ldflags
# fi
#
# make clean && make YK_BUILD_TYPE=release
#
# cd tests
#
# LUA=../src/lua
#
# for serialise in 0 1; do
#     for test in api bwcoercion closure code coroutine events \
#         gengc pm tpack tracegc utf8 vararg; do
#         echo "### YKD_SERIALISE_COMPILATION=$serialise $test.lua ###"
#         YKD_SERIALISE_COMPILATION=$serialise ${LUA} ${test}.lua
#     done
# done
