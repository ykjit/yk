-- Run-time:
--   env-var: YK_HOT_THRESHOLD=2
--   env-var: YKD_LOG=4
--   env-var: YKD_LOG_IR=debugstrs
--   env-var: YKD_SERIALISE_COMPILATION=1
--   stderr:
--     8
--     7
--     6
--     yk-tracing: start-tracing: recursive_function_indirect.lua:66: GETTABUP
--     5
--     yk-tracing: stop-tracing: recursive_function_indirect.lua:66: GETTABUP
--     --- Begin debugstrs: recursive_function_indirect.lua:66: GETTABUP ---
--     ; {
--     ;   "trid": "0",
--     ;   "start": {
--     ;     "kind": "ControlPoint"
--     ;   },
--     ;   "end": {
--     ;     "kind": "Loop"
--     ;   }
--     ; }
--     recursive_function_indirect.lua:66: GETTABUP
--     recursive_function_indirect.lua:66: GETFIELD
--     recursive_function_indirect.lua:66: SELF
--     recursive_function_indirect.lua:66: GETTABUP
--     recursive_function_indirect.lua:66: MOVE
--     recursive_function_indirect.lua:66: CALL
--     recursive_function_indirect.lua:66: LOADK
--     recursive_function_indirect.lua:66: CALL
--     recursive_function_indirect.lua:67: GTI
--     recursive_function_indirect.lua:68: GETTABUP
--     recursive_function_indirect.lua:68: ADDI
--     recursive_function_indirect.lua:68: CALL
--     recursive_function_indirect.lua:73: GETTABUP
--     recursive_function_indirect.lua:73: MOVE
--     recursive_function_indirect.lua:73: TAILCALL
--     --- End debugstrs ---
--     4
--     yk-tracing: start-tracing: recursive_function_indirect.lua:73: GETTABUP
--     yk-tracing: stop-tracing: recursive_function_indirect.lua:66: GETTABUP
--     --- Begin debugstrs: recursive_function_indirect.lua:73: GETTABUP ---
--     ; {
--     ;   "trid": "1",
--     ;   "start": {
--     ;     "kind": "ControlPoint"
--     ;   },
--     ;   "end": {
--     ;     "kind": "Coupler",
--     ;     "tgt_trid": "0"
--     ;   }
--     ; }
--     recursive_function_indirect.lua:73: GETTABUP
--     recursive_function_indirect.lua:73: MOVE
--     recursive_function_indirect.lua:73: TAILCALL
--     --- End debugstrs ---
--     3
--     yk-execution: enter-jit-code: recursive_function_indirect.lua:73: GETTABUP
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
