-- ignore-if: test "$YKB_TRACER" = "swt"
-- Run-time:
--   env-var: YK_HOT_THRESHOLD=3
--   env-var: YKD_LOG=4
--   env-var: YKD_LOG_IR=trace-kind
--   env-var: YKD_SERIALISE_COMPILATION=1
--   stderr:
--     6
--     5
--     4
--     yk-jit-event: start-tracing
--     3
--     yk-jit-event: stop-tracing
--     --- trace-kind header ---
--     2
--     yk-jit-event: enter-jit-code
--     1
--     0
--     yk-jit-event: deoptimise
--     exit

function f(x)
  io.stderr:write(tostring(x), "\n")
  if x > 0 then
    f(x - 1)
  end
end

f(6)
io.stderr:write("exit\n")
