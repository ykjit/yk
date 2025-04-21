-- ignore-if: test "$YKB_TRACER" = "swt"
-- Run-time:
--   env-var: YK_HOT_THRESHOLD=3
--   env-var: YKD_LOG=4
--   env-var: YKD_LOG_IR=trace-kind
--   env-var: YKD_SERIALISE_COMPILATION=1
--   stderr:
--     0
--     1
--     2
--     yk-jit-event: start-tracing: for_loop.lua:23: FORLOOP
--     3
--     yk-jit-event: stop-tracing: ...
--     --- trace-kind header ---
--     4
--     yk-jit-event: enter-jit-code: for_loop.lua:23: FORLOOP
--     5
--     6
--     yk-jit-event: deoptimise
--     exit

local x = 0
for _ = 0, 6 do
  io.stderr:write(tostring(x), "\n")
  x = x + 1
end
io.stderr:write("exit\n")
