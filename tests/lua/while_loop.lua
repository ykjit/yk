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
--     yk-jit-event: start-tracing: while_loop.lua:25: JMP
--     3
--     yk-jit-event: stop-tracing: ...
--     --- trace-kind header ---
--     4
--     yk-jit-event: enter-jit-code: while_loop.lua:25: JMP
--     5
--     6
--     yk-jit-event: deoptimise
--     exit

local x = 0
while x <= 6 do
  io.stderr:write(tostring(x), "\n")
  x = x + 1
end
io.stderr:write("exit\n")
