YKB_SWT_MODCLONE=1 YKB_TRACER=swt cargo build
cd /home/pd/yk-benchmarks/suites/awfy/Lua
RUST_BACKTRACE=1  YKD_SERIALISE_COMPILATION=1  ~/yklua-fork/src/lua ./harness.lua richards 1 100|& tee ~/yk-fork/out.txt 
cd -
