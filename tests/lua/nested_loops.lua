-- ignore-if: test "$YKB_TRACER" = "swt"
-- Run-time:
--   env-var: YKD_LOG=4
--   env-var: YKD_LOG_IR=trace-kind
--   env-var: YKD_SERIALISE_COMPILATION=1
--   stderr:
--     yk-jit-event: start-tracing
--     yk-jit-event: stop-tracing
--     --- trace-kind header ---
--     yk-jit-event: enter-jit-code
--     yk-jit-event: deoptimise
--     yk-jit-event: enter-jit-code
--     yk-jit-event: deoptimise
--     yk-jit-event: enter-jit-code
--     yk-jit-event: deoptimise
--     yk-jit-event: enter-jit-code
--     yk-jit-event: deoptimise
--     yk-jit-event: enter-jit-code
--     yk-jit-event: deoptimise
--     yk-jit-event: start-side-tracing
--     yk-jit-event: stop-tracing
--     --- trace-kind side-trace ---
--     yk-jit-event: enter-jit-code
--     yk-jit-event: deoptimise
--   stdout:
--     251502

local x = 0
for _ = 0, 500 do
  x = x + 1
  for _ = 0, 500 do
    x = x + 1
  end
end
print(x)
