-- Run-time:
--   env-var: YK_HOT_THRESHOLD=2
--   env-var: YKD_LOG=4
--   env-var: YKD_LOG_IR=debugstrs
--   env-var: YKD_SERIALISE_COMPILATION=1
--   stderr:
--     8
--     7
--     6
--     yk-tracing: start-tracing: recursive_function_indirect.lua:47: GETTABUP
--     5
--     yk-tracing: stop-tracing: recursive_function_indirect.lua:47: GETTABUP
--     --- Begin debugstrs: header: recursive_function_indirect.lua:47: GETTABUP ---
--       recursive_function_indirect.lua:47: GETTABUP
--       recursive_function_indirect.lua:47: GETFIELD
--       recursive_function_indirect.lua:47: SELF
--       recursive_function_indirect.lua:47: GETTABUP
--       recursive_function_indirect.lua:47: MOVE
--       recursive_function_indirect.lua:47: CALL
--       recursive_function_indirect.lua:47: LOADK
--       recursive_function_indirect.lua:47: CALL
--       recursive_function_indirect.lua:48: GTI
--       recursive_function_indirect.lua:49: GETTABUP
--       recursive_function_indirect.lua:49: ADDI
--       recursive_function_indirect.lua:49: CALL
--       recursive_function_indirect.lua:54: GETTABUP
--       recursive_function_indirect.lua:54: MOVE
--       recursive_function_indirect.lua:54: TAILCALL
--     --- End debugstrs ---
--     4
--     yk-tracing: start-tracing: recursive_function_indirect.lua:54: GETTABUP
--     yk-tracing: stop-tracing: recursive_function_indirect.lua:47: GETTABUP
--     --- Begin debugstrs: connector: recursive_function_indirect.lua:54: GETTABUP ---
--       recursive_function_indirect.lua:54: GETTABUP
--       recursive_function_indirect.lua:54: MOVE
--       recursive_function_indirect.lua:54: TAILCALL
--     --- End debugstrs ---
--     3
--     yk-execution: enter-jit-code: recursive_function_indirect.lua:54: GETTABUP
--     2
--     1
--     0
--     yk-execution: deoptimise ...
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
