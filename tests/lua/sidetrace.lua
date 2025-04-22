-- ignore-if: test "$YKB_TRACER" = "swt"
-- Run-time:
--   env-var: YK_HOT_THRESHOLD=3
--   env-var: YK_SIDETRACE_THRESHOLD=2
--   env-var: YKD_LOG=4
--   env-var: YKD_LOG_IR=trace-kind
--   env-var: YKD_SERIALISE_COMPILATION=1
--   stderr:
--     <0
--     <1
--     <2
--     yk-jit-event: start-tracing
--     <3
--     yk-jit-event: stop-tracing
--     --- trace-kind header ---
--     <4
--     yk-jit-event: enter-jit-code
--     yk-jit-event: deoptimise
--     >=5
--     yk-jit-event: enter-jit-code
--     yk-jit-event: deoptimise
--     yk-jit-event: start-side-tracing
--     >=6
--     yk-jit-event: stop-tracing
--     --- trace-kind side-trace ---
--     >=7
--     yk-jit-event: enter-jit-code
--     >=8
--     >=9
--     >=10
--     yk-jit-event: deoptimise
--     exit

for i = 0, 10 do
  if i < 5 then
    io.stderr:write("<", tostring(i), "\n")
  else
    io.stderr:write(">=", tostring(i), "\n")
  end
end
io.stderr:write("exit\n")
