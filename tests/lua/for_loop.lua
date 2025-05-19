-- ignore-if: test "$YKB_TRACER" = "swt"
-- Run-time:
--   env-var: YK_HOT_THRESHOLD=3
--   env-var: YKD_LOG=4
--   env-var: YKD_LOG_IR=debugstrs
--   env-var: YKD_SERIALISE_COMPILATION=1
--   stderr:
--     0
--     1
--     2
--     yk-tracing: start-tracing: for_loop.lua:35: GETTABUP
--     3
--     yk-tracing: stop-tracing: ...
--     --- Begin debugstrs: header: for_loop.lua:35: GETTABUP ---
--       for_loop.lua:35: GETFIELD
--       for_loop.lua:35: SELF
--       for_loop.lua:35: GETTABUP
--       for_loop.lua:35: MOVE
--       for_loop.lua:35: CALL
--       for_loop.lua:35: LOADK
--       for_loop.lua:35: CALL
--       for_loop.lua:36: ADDI
--       for_loop.lua:34: FORLOOP
--       for_loop.lua:35: GETTABUP
--     --- End debugstrs ---
--     4
--     yk-execution: enter-jit-code: for_loop.lua:35: GETTABUP
--     5
--     6
--     yk-execution: deoptimise
--     exit

local x = 0
for _ = 0, 6 do
  io.stderr:write(tostring(x), "\n")
  x = x + 1
end
io.stderr:write("exit\n")
