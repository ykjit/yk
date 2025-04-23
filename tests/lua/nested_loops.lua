-- ignore-if: test "$YKB_TRACER" = "swt"
-- Run-time:
--   env-var: YKD_LOG=4
--   env-var: YKD_LOG_IR=debugstrs
--   env-var: YKD_SERIALISE_COMPILATION=1
--   stderr:
--     yk-jit-event: start-tracing: nested_loops.lua:42: ADDI
--     yk-jit-event: stop-tracing: nested_loops.lua:42: ADDI
--     --- Begin debugstrs: header: nested_loops.lua:42: ADDI ---
--       nested_loops.lua:41: FORLOOP
--       nested_loops.lua:42: ADDI
--     --- End debugstrs ---
--     yk-jit-event: enter-jit-code: nested_loops.lua:42: ADDI
--     yk-jit-event: deoptimise
--     yk-jit-event: enter-jit-code: nested_loops.lua:42: ADDI
--     yk-jit-event: deoptimise
--     yk-jit-event: enter-jit-code: nested_loops.lua:42: ADDI
--     yk-jit-event: deoptimise
--     yk-jit-event: enter-jit-code: nested_loops.lua:42: ADDI
--     yk-jit-event: deoptimise
--     yk-jit-event: enter-jit-code: nested_loops.lua:42: ADDI
--     yk-jit-event: deoptimise
--     yk-jit-event: start-side-tracing: nested_loops.lua:42: ADDI
--     yk-jit-event: stop-tracing: nested_loops.lua:42: ADDI
--     --- Begin debugstrs: side-trace: nested_loops.lua:42: ADDI ---
--       nested_loops.lua:39: FORLOOP
--       nested_loops.lua:40: ADDI
--       nested_loops.lua:41: LOADI
--       nested_loops.lua:41: LOADI
--       nested_loops.lua:41: LOADI
--       nested_loops.lua:41: FORPREP
--       nested_loops.lua:42: ADDI
--     --- End debugstrs ---
--     yk-jit-event: enter-jit-code: nested_loops.lua:42: ADDI
--     yk-jit-event: deoptimise
--     251502

local x = 0
for _ = 0, 500 do
  x = x + 1
  for _ = 0, 500 do
    x = x + 1
  end
end
io.stderr:write(x)
