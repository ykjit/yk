-- ignore-if: test "$YKB_TRACER" = "swt"
-- Run-time:
--   env-var: YK_HOT_THRESHOLD=2
--   env-var: YKD_LOG=4
--   env-var: YKD_LOG_IR=debugstrs
--   env-var: YKD_SERIALISE_COMPILATION=1
--   stderr:
--     8
--     7
--     6
--     yk-jit-event: start-tracing: recursive_function_indirect.lua:48: GETTABUP
--     5
--     yk-jit-event: stop-tracing: ...
--     --- Begin debugstrs: header: recursive_function_indirect.lua:48: GETTABUP ---
--       recursive_function_indirect.lua:48: GETFIELD
--       recursive_function_indirect.lua:48: SELF
--       recursive_function_indirect.lua:48: GETTABUP
--       recursive_function_indirect.lua:48: MOVE
--       recursive_function_indirect.lua:48: CALL
--       recursive_function_indirect.lua:48: LOADK
--       recursive_function_indirect.lua:48: CALL
--       recursive_function_indirect.lua:49: GTI
--       recursive_function_indirect.lua:50: GETTABUP
--       recursive_function_indirect.lua:50: ADDI
--       recursive_function_indirect.lua:50: CALL
--       recursive_function_indirect.lua:55: GETTABUP
--       recursive_function_indirect.lua:55: MOVE
--       recursive_function_indirect.lua:55: TAILCALL
--       recursive_function_indirect.lua:48: GETTABUP
--     --- End debugstrs ---
--     4
--     yk-jit-event: start-tracing: recursive_function_indirect.lua:55: GETTABUP
--     yk-jit-event: stop-tracing: recursive_function_indirect.lua:48: GETTABUP
--     --- Begin debugstrs: connector: recursive_function_indirect.lua:55: GETTABUP ---
--       recursive_function_indirect.lua:55: MOVE
--       recursive_function_indirect.lua:55: TAILCALL
--       recursive_function_indirect.lua:48: GETTABUP
--     --- End debugstrs ---
--     3
--     yk-jit-event: enter-jit-code: recursive_function_indirect.lua:55: GETTABUP
--     2
--     1
--     0
--     yk-jit-event: deoptimise
--     exit

function f(x)
  io.stderr:write(tostring(x), "\n")
  if x > 0 then
    g(x - 1)
  end
end

function g(x)
  return f(x)
end

f(8)
io.stderr:write("exit\n")
