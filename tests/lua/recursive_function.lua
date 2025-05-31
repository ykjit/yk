-- ignore-if: test "$YKB_TRACER" = "swt"
-- Run-time:
--   env-var: YK_HOT_THRESHOLD=2
--   env-var: YKD_LOG=4
--   env-var: YKD_LOG_IR=debugstrs
--   env-var: YKD_SERIALISE_COMPILATION=1
--   stderr:
--     6
--     5
--     4
--     yk-tracing: start-tracing: recursive_function.lua:36: GETTABUP
--     3
--     yk-tracing: stop-tracing: recursive_function.lua:36: GETTABUP
--     --- Begin debugstrs: header: recursive_function.lua:36: GETTABUP ---
--       recursive_function.lua:36: GETTABUP
--       recursive_function.lua:36: GETFIELD
--       recursive_function.lua:36: SELF
--       recursive_function.lua:36: GETTABUP
--       recursive_function.lua:36: MOVE
--       recursive_function.lua:36: CALL
--       recursive_function.lua:36: LOADK
--       recursive_function.lua:36: CALL
--       recursive_function.lua:37: GTI
--       recursive_function.lua:38: GETTABUP
--       recursive_function.lua:38: ADDI
--       recursive_function.lua:38: CALL
--     --- End debugstrs ---
--     2
--     yk-execution: enter-jit-code: recursive_function.lua:36: GETTABUP
--     1
--     0
--     yk-execution: deoptimise
--     exit

function f(x)
  io.stderr:write(tostring(x), "\n")
  if x > 0 then
    f(x - 1)
  end
end

f(6)
io.stderr:write("exit\n")
