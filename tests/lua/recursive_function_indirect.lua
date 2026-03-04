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

-- Run-time:
--   env-var: YK_HOT_THRESHOLD=2
--   env-var: YKD_LOG=4
--   env-var: YKD_LOG_IR=debugstrs
--   env-var: YKD_SERIALISE_COMPILATION=1
--   stderr:
--     8
--     7
--     6
--     yk-tracing: start-tracing: recursive_function_indirect.lua:2: GETTABUP
--     5
--     yk-tracing: stop-tracing: recursive_function_indirect.lua:2: GETTABUP
--     --- Begin debugstrs: recursive_function_indirect.lua:2: GETTABUP ---
--     ; {
--     ;   "trid": "0",
--     ;   "start": {
--     ;     "kind": "ControlPoint"
--     ;   },
--     ;   "end": {
--     ;     "kind": "Loop"
--     ;   }
--     ; }
--     recursive_function_indirect.lua:2: GETTABUP
--     recursive_function_indirect.lua:2: GETFIELD
--     recursive_function_indirect.lua:2: SELF
--     recursive_function_indirect.lua:2: GETTABUP
--     recursive_function_indirect.lua:2: MOVE
--     recursive_function_indirect.lua:2: CALL
--     recursive_function_indirect.lua:2: LOADK
--     recursive_function_indirect.lua:2: CALL
--     recursive_function_indirect.lua:3: GTI
--     recursive_function_indirect.lua:4: GETTABUP
--     recursive_function_indirect.lua:4: ADDI
--     recursive_function_indirect.lua:4: CALL
--     recursive_function_indirect.lua:9: GETTABUP
--     recursive_function_indirect.lua:9: MOVE
--     recursive_function_indirect.lua:9: TAILCALL
--     --- End debugstrs ---
--     4
--     yk-tracing: start-tracing: recursive_function_indirect.lua:9: GETTABUP
--     yk-tracing: stop-tracing: recursive_function_indirect.lua:2: GETTABUP
--     --- Begin debugstrs: recursive_function_indirect.lua:9: GETTABUP ---
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
--     recursive_function_indirect.lua:9: GETTABUP
--     recursive_function_indirect.lua:9: MOVE
--     recursive_function_indirect.lua:9: TAILCALL
--     --- End debugstrs ---
--     3
--     yk-execution: enter-jit-code: recursive_function_indirect.lua:9: GETTABUP
--     2
--     1
--     0
--     yk-execution: deoptimise ...
--     exit
