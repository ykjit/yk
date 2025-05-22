-- ignore-if: test "$YKB_TRACER" = "swt"
-- Run-time:
--   env-var: YKD_LOG=4
--   env-var: YKD_LOG_IR=debugstrs
--   env-var: YKD_SERIALISE_COMPILATION=1
--   stderr:
--     yk-tracing: start-tracing: nested_loops.lua:46: ADDI
--     yk-tracing: stop-tracing: nested_loops.lua:46: ADDI
--     --- Begin debugstrs: header: nested_loops.lua:46: ADDI ---
--       nested_loops.lua:45: FORLOOP
--       nested_loops.lua:46: ADDI
--     --- End debugstrs ---
--     yk-execution: enter-jit-code: nested_loops.lua:46: ADDI
--     yk-execution: deoptimise
--     yk-execution: enter-jit-code: nested_loops.lua:46: ADDI
--     yk-execution: deoptimise
--     yk-execution: enter-jit-code: nested_loops.lua:46: ADDI
--     yk-execution: deoptimise
--     yk-execution: enter-jit-code: nested_loops.lua:46: ADDI
--     yk-execution: deoptimise
--     yk-execution: enter-jit-code: nested_loops.lua:46: ADDI
--     yk-execution: deoptimise
--     yk-tracing: start-side-tracing: nested_loops.lua:46: ADDI
--     yk-tracing: stop-tracing: nested_loops.lua:44: ADDI
--     yk-tracing: start-tracing: nested_loops.lua:44: ADDI
--     yk-tracing: stop-tracing: nested_loops.lua:46: ADDI
--     --- Begin debugstrs: connector: nested_loops.lua:44: ADDI ---
--       nested_loops.lua:45: LOADI
--       nested_loops.lua:45: LOADI
--       nested_loops.lua:45: LOADI
--       nested_loops.lua:45: FORPREP
--       nested_loops.lua:46: ADDI
--     --- End debugstrs ---
--     --- Begin debugstrs: side-trace: nested_loops.lua:46: ADDI ---
--       nested_loops.lua:43: FORLOOP
--       nested_loops.lua:44: ADDI
--     --- End debugstrs ---
--     yk-execution: enter-jit-code: nested_loops.lua:46: ADDI
--     yk-execution: deoptimise
--     251502

local x = 0
for _ = 0, 500 do
  x = x + 1
  for _ = 0, 500 do
    x = x + 1
  end
end
io.stderr:write(x)
